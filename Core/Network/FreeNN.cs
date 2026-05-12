using System.Text;

namespace EliasFM.ArtificialNeuralNet.Core.Network;

public abstract partial class FreeNN : INetwork {

	public long TotalNeuronCount => HiddenNeuronCount + OutputLength;

	public abstract int HiddenNeuronCount { get; }

	public double[] Predict(double[] networkInput) {
		throw new NotImplementedException();
	}

	public abstract int InputLength { get; }

	public abstract int OutputLength { get; }


}


