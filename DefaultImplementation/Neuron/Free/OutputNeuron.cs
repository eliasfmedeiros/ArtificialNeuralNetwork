using EliasFM.ArtificialNeuralNet.CommonImplementation.Neuron.Free.Auxiliary;
using EliasFM.ArtificialNeuralNet.Core;
using EliasFM.ArtificialNeuralNet.Core.Network;

namespace EliasFM.ArtificialNeuralNet.CommonImplementation.Neuron.Free;

public abstract class OutputNeuron : FreeNN.InputDescribedNeuron {
	public SignalChannel[] ChannelArr { get; private set; }
	private double bias;

	public OutputNeuron(IActivation activation, IInputDescriptor[] input, Random? weightStarter = null) {
		ChannelArr = new SignalChannel[input.Length];
		for (int i = 0; i < input.Length; i++) ChannelArr[i] = new SignalChannel(input[i], weightStarter);
		bias = DefaultStartingWeight(weightStarter);
	}
	public OutputNeuron(IActivation activator, SignalChannel[] input, double bias) {
		ChannelArr = input;
		this.bias = bias;
	}

	public override int InputLength => ChannelArr.Length;

	public override double GetWeightValue(int index) =>
		index == InputLength ? GetBiasValue() : ChannelArr[index].Weight;

	public override IInputDescriptor DescribeInput(int index) => ChannelArr[index].InputDescriptor;

	public override double GetBiasValue() => bias;

	#region subclasses
	//TODO
	#endregion

}
