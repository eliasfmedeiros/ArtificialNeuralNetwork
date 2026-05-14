using EliasFM.ArtificialNeuralNet.Core;

namespace EliasFM.ArtificialNeuralNet.CommonImplementation.Function;

public class Tanh : IActivation { // Mantido como class por consistência com as demais ativações e possível uso futuro de herança/polimorfismo.

	public static readonly Tanh Shared = new();

	public static double Function(double x) => Math.Tanh(x);

	public double Apply(double input) => Function(input);

	public override string ToString() => GetType().Name;
}
