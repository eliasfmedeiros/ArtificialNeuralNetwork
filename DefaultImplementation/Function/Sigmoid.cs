using EliasFM.ArtificialNeuralNet.Core;

namespace EliasFM.ArtificialNeuralNet.CommonImplementation.Function;

public class Sigmoid : IActivation { // Mantido como class por consistência com as demais ativações e possível uso futuro de herança/polimorfismo.

	public static readonly Sigmoid Shared = new();

	public static double Function(double x) => 1.0 / (1.0 + Math.Exp(-x));

	public double Apply(double input) => Function(input);

	public override string ToString() => GetType().Name;
}
