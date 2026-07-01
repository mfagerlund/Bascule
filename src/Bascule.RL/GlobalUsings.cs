// The RL core used to live in namespace Tensotron.Rl — a child of the engine's Tensotron namespace — so it
// saw the engine's root types (Tensor, TensorOps, Linear, Sequential, Adam, Activation, ...) for free via
// parent-namespace lookup, with no explicit using. Renamed to Bascule.RL, that implicit access is gone;
// restore it once for the whole project. This is the RL core's sole dependency on the Tensotron engine.
global using Tensotron;
