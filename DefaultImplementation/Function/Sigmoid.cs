using EliasFM.ArtificialNeuralNet.Core;

namespace EliasFM.ArtificialNeuralNet.CommonImplementation.Function;

public class Sigmoid : IActivation {
	public static double Function(double x) => 1.0 / (1.0 + Math.Exp(-x));

	public double Apply(double input) => Function(input);

	public override string ToString() => GetType().Name;

	public static readonly Sigmoid Shared = new();
}
