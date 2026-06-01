using EliasFM.ArtificialNeuralNet.Core.Network;
using static EliasFM.ArtificialNeuralNet.Core.Network.FreeNN.InputDescribedNeuron;

namespace EliasFM.ArtificialNeuralNet.CommonImplementation.Neuron.Free.Auxiliary;

public struct SignalChannel {
	public SignalChannel(IInputDescriptor inputDescriptor, Random? weightStarter = null) {
		InputDescriptor = inputDescriptor;
		Weight = Core.Neuron.DefaultStartingWeight(weightStarter);
	}
	public SignalChannel(FreeNN.InputSourceType sourceType, int indexInSource, Random? weightStarter = null) {
		InputDescriptor = new InputDescriptor(sourceType, indexInSource);
		Weight = Core.Neuron.DefaultStartingWeight(weightStarter);
	}
	public SignalChannel(IInputDescriptor inputDescriptor, double weight) {
		InputDescriptor = inputDescriptor;
		Weight = weight;
	}
	public SignalChannel(FreeNN.InputSourceType sourceType, int indexInSource, double weight) {
		InputDescriptor = new InputDescriptor(sourceType, indexInSource);
		Weight = weight;
	}

	public IInputDescriptor InputDescriptor { get; set; }
	public double Weight { get; set; }
}



