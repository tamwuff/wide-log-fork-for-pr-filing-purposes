//! The future type behind the generated async `scope()` family.
//!
//! **Feature gate:** compiled only with the `tokio` feature.
//!
//! The generated `scope()` functions used to be `async fn`s. An `async fn`
//! that awaits a future it took as an argument stores that future twice:
//! once as the captured argument and once inside the awaited value, and
//! the compiler does not let the two share bytes. [`ScopeFuture`] is a
//! hand-written future that stores the wrapped future once.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, ready};

use pin_project_lite::pin_project;

pin_project! {
    /// Polls `future` to completion and drops `guard` before it returns
    /// the output.
    ///
    /// The generated `scope()` family builds the event guard when it is
    /// called and returns this future. `future` is the user's future
    /// inside the task-local scope(s); `guard` owns the event and emits it
    /// on drop.
    ///
    /// Drop order:
    ///
    /// - On completion, `guard` drops inside the `poll` call that returns
    ///   `Poll::Ready`, after `future` has finished.
    /// - On cancellation (the `ScopeFuture` is dropped before completion),
    ///   `future` drops first and `guard` drops second, so the event is
    ///   still alive while the wrapped future's destructors run.
    ///
    /// # Panics
    ///
    /// `poll` panics if it is called again after returning `Poll::Ready`.
    /// This matches compiler-generated async futures and, importantly,
    /// rejects the poll before re-entering the completed task-local future.
    #[must_use = "futures do nothing unless you `.await` or poll them"]
    pub struct ScopeFuture<Fut, G> {
        // Declared before `guard`: fields drop in declaration order.
        #[pin]
        future: Fut,
        guard: Option<G>,
    }
}

impl<Fut, G> ScopeFuture<Fut, G> {
    /// Wraps `future` so that `guard` lives until `future` completes or
    /// the returned future is dropped.
    #[inline]
    pub fn new(future: Fut, guard: G) -> Self {
        Self {
            future,
            guard: Some(guard),
        }
    }
}

impl<Fut: Future, G> Future for ScopeFuture<Fut, G> {
    type Output = Fut::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        if this.guard.is_none() {
            panic!("`ScopeFuture` polled after completion");
        }
        let output = ready!(this.future.poll(cx));
        // Drop the guard now, not when the caller drops this future, so
        // the event is emitted before the output is returned. `take`
        // leaves the completed marker in place even if the guard panics
        // while it is being dropped.
        drop(this.guard.take());
        Poll::Ready(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::pin::pin;
    use std::rc::Rc;
    use std::task::Waker;

    type DropLog = Rc<RefCell<Vec<&'static str>>>;

    /// Records `name` in the log when dropped.
    struct RecordDrop {
        name: &'static str,
        log: DropLog,
    }

    impl Drop for RecordDrop {
        fn drop(&mut self) {
            self.log.borrow_mut().push(self.name);
        }
    }

    /// Pending on the first poll, ready with `7` on the second. Records
    /// its own drop as `"future"`.
    struct PendingOnce {
        polled: bool,
        _on_drop: RecordDrop,
    }

    impl Future for PendingOnce {
        type Output = u32;

        fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<u32> {
            if self.polled {
                Poll::Ready(7)
            } else {
                self.polled = true;
                Poll::Pending
            }
        }
    }

    /// Ready immediately and counts every call to `poll`.
    struct CountPolls {
        polls: Rc<Cell<u32>>,
    }

    impl Future for CountPolls {
        type Output = u32;

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<u32> {
            self.polls.set(self.polls.get() + 1);
            Poll::Ready(7)
        }
    }

    fn scope_future(log: &DropLog) -> ScopeFuture<PendingOnce, RecordDrop> {
        let future = PendingOnce {
            polled: false,
            _on_drop: RecordDrop {
                name: "future",
                log: log.clone(),
            },
        };
        let guard = RecordDrop {
            name: "guard",
            log: log.clone(),
        };
        ScopeFuture::new(future, guard)
    }

    #[test]
    fn guard_drops_when_the_future_completes() {
        let log = DropLog::default();
        let mut fut = pin!(scope_future(&log));
        let mut cx = Context::from_waker(Waker::noop());

        assert_eq!(fut.as_mut().poll(&mut cx), Poll::Pending);
        assert!(log.borrow().is_empty(), "the guard lives while pending");

        assert_eq!(fut.as_mut().poll(&mut cx), Poll::Ready(7));
        assert_eq!(
            *log.borrow(),
            vec!["guard"],
            "the guard drops inside the completing poll"
        );
    }

    #[test]
    fn cancellation_drops_the_future_before_the_guard() {
        let log = DropLog::default();
        {
            let mut fut = pin!(scope_future(&log));
            let mut cx = Context::from_waker(Waker::noop());
            assert_eq!(fut.as_mut().poll(&mut cx), Poll::Pending);
        }
        assert_eq!(*log.borrow(), vec!["future", "guard"]);
    }

    #[test]
    fn unpolled_drop_drops_the_future_before_the_guard() {
        let log = DropLog::default();
        drop(scope_future(&log));
        assert_eq!(*log.borrow(), vec!["future", "guard"]);
    }

    #[test]
    fn poll_after_completion_panics_before_polling_the_inner_future() {
        let polls = Rc::new(Cell::new(0));
        let log = DropLog::default();
        let guard = RecordDrop {
            name: "guard",
            log: log.clone(),
        };
        let mut future = Box::pin(ScopeFuture::new(
            CountPolls {
                polls: polls.clone(),
            },
            guard,
        ));
        let mut cx = Context::from_waker(Waker::noop());

        assert_eq!(future.as_mut().poll(&mut cx), Poll::Ready(7));
        assert_eq!(polls.get(), 1);
        assert_eq!(*log.borrow(), vec!["guard"]);

        let repoll = catch_unwind(AssertUnwindSafe(|| future.as_mut().poll(&mut cx)));
        assert!(repoll.is_err());
        assert_eq!(polls.get(), 1, "the completed inner future was repolled");
    }
}
