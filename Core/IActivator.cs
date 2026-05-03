namespace EliasFM.ArtificialNeuralNet.Core;

public interface IActivator {
	double ApplyActivationFunction(double input);

	public struct FromFunction(Func<double, double> function) : IActivator {
		public readonly double ApplyActivationFunction(double input) {
			return function(input);
		}

	}
}
