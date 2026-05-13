namespace EliasFM.ArtificialNeuralNet.Core;

public interface IActivation {

	double Apply(double input);

	public struct FromFunction(Func<double, double> function) : IActivation {
		public readonly double Apply(double input) {
			return function(input);
		}
	}
}
