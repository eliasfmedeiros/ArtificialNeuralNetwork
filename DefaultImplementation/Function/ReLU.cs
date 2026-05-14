using EliasFM.ArtificialNeuralNet.Core;

namespace EliasFM.ArtificialNeuralNet.CommonImplementation.Function;

public class ReLU : IActivation {

	public static readonly ReLU Shared = new();

	public virtual double GetLeak() => 0;

	public static double Function(double x, double leak = 0) => x < 0 ? leak * x : x;

	public double Apply(double input) => Function(input, GetLeak());

	public override string ToString() => GetType().Name;

	public class Leaky(double leak) : ReLU {
		public override double GetLeak() => leak;
		public virtual void SetLeak(double value) => leak = value;
	}
}
