using EliasFM.ArtificialNeuralNet.Core;

namespace EliasFM.ArtificialNeuralNet.CommonImplementation.Function;

public class RectifiedTanh : IActivation { // ReHTU

	public static readonly RectifiedTanh Shared = new();

	public virtual double GetLeak() => 0;
	public virtual double GetLinearity() => 0;

	public static double Function(double x, double leak = 0, double linearity = 0) {
		double tanh = Math.Tanh(x);
		return ReLU.Function((linearity * (x - tanh)) + tanh, leak);
	}

	public double Apply(double input) => Function(input, GetLeak(), GetLinearity());

	public override string ToString() {
		return GetType().Name + '{' + $"Leak:{GetLeak()}" + "," + $"Linearity:{GetLinearity()}" + '}';
	}

	public class LeakyLinearizable(double linearity, double leak) : RectifiedTanh {
		public override double GetLinearity() => linearity;
		public virtual void SetLinearity(double value) => linearity = value;
		public override double GetLeak() => leak;
		public virtual void SetLeak(double value) => leak = value;
	}
	public class Linearizable(double linearity = 0) : RectifiedTanh {
		public override double GetLinearity() => linearity;
		public virtual void SetLinearity(double value) => linearity = value;
	}
	public class Leaky(double leak = 0) : RectifiedTanh {
		public override double GetLeak() => leak;
		public virtual void SetLeak(double value) => leak = value;
	}
}
