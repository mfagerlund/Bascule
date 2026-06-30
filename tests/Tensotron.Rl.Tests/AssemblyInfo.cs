using Xunit;

// The Tensotron engine's TensorRuntime is a process-wide, single-threaded singleton: calling tensor
// ops from multiple threads concurrently is unsupported and corrupts shared state (allocator pool,
// caches, FlushEvery). xUnit parallelizes test collections by default, so two training tests running
// at once race on that singleton and intermittently fail. Serialize the whole assembly — these are
// fast tensor tests, so the cost is negligible and correctness is non-negotiable.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
