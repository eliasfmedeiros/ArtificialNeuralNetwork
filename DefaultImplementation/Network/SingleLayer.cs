using EliasFM.ArtificialNeuralNet.CommonImplementation.Neuron;
using EliasFM.ArtificialNeuralNet.Core;
using static EliasFM.ArtificialNeuralNet.CommonImplementation.Network.Tool;
using NeuronBase = EliasFM.ArtificialNeuralNet.Core.Neuron;
using SingleLayerBase = EliasFM.ArtificialNeuralNet.Core.Network.SingleLayer;

namespace EliasFM.ArtificialNeuralNet.CommonImplementation.Network;

public class SingleLayer : SingleLayerBase {
	private readonly NeuronBase[] neuronArr;
	public override int NeuronCount => neuronArr.Length;
	public override int InputLength { get; }

	/*
	public SingleLayer(int inputs, NeuronBase[] neuronArr) {
		InputLength = inputs;
		this.neuronArr = new NeuronBase[neuronArr.Length];
		for (int i = 0; i < neuronArr.Length; i++) {
			if (neuronArr[i].InputLength != InputLength) throw new ArgumentException("Inconsistent neuron input dimensionality breaks single-layer vector coherence.");
			this.neuronArr[i] = neuronArr[i];
		}
	}
	*/
	public SingleLayer(int inputs, IActivation[] activators) {
		InputLength = inputs;
		neuronArr = new NeuronBase[activators.Length];
		for (int i = 0; i < activators.Length; i++)
			neuronArr[i] = new LayerNeuron.CustomActivated(inputs, activators[i]);
	}
	public SingleLayer(int inputs, IActivation activator, int outputs) {
		InputLength = inputs;
		neuronArr = new NeuronBase[outputs];
		while (outputs-- > 0)
			neuronArr[outputs] = new LayerNeuron.CustomActivated(inputs, activator);
	}

	public override NeuronBase NeuronAt(int index) => neuronArr[index];
	public LayerNeuron LayerNeuronAt(int index) => (LayerNeuron)NeuronAt(index);

	/*
	public virtual object ReplaceNeuron(int neuronIndex, NeuronBase newNeuron) {
		if (newNeuron.InputLength != InputLength)
			throw new ArgumentException(
				$"Incompatible input length: expected {InputLength}, but got {newNeuron.InputLength}.",
				nameof(newNeuron)
			);
		NeuronBase replaced = neuronArr[neuronIndex];
		neuronArr[neuronIndex] = newNeuron;
		return replaced;
	}
	*/
}
