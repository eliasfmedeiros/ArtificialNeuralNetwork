namespace EliasFM.ArtificialNeuralNet.Core.Network;

public interface INetwork : INeural {
	public double[] Predict(double[] input);
}
