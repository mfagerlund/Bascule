// Tests moved out of Tensotron.Rl.Tests (a child of the engine's Tensotron namespace) into Bascule.RL.Tests,
// losing implicit parent-namespace access to engine types (Init, Tensor, ...). Restore it for the project.
global using Tensotron;
