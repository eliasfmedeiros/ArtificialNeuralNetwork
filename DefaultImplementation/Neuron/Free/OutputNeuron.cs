using EliasFM.ArtificialNeuralNet.CommonImplementation.Neuron.Free.Auxiliary;
using EliasFM.ArtificialNeuralNet.Core;
using EliasFM.ArtificialNeuralNet.Core.Network;

namespace EliasFM.ArtificialNeuralNet.CommonImplementation.Neuron.Free;

public abstract class OutputNeuron : FreeNN.InputDescribedNeuron {
	public SignalChannel[] ChannelArr { get; private set; }
	private double bias;

	public OutputNeuron(IInputDescriptor[] input, Random? weightStarter = null) {
		//, double precomputedOutput = 0) : base(precomputedOutput) {
		ChannelArr = new SignalChannel[input.Length];
		for (int i = 0; i < input.Length; i++) ChannelArr[i] = new SignalChannel(input[i], weightStarter);
		bias = DefaultStartingWeight(weightStarter);
	}
	public OutputNeuron(SignalChannel[] input, double bias) {
		//, double precomputedOutput = 0) : base(precomputedOutput) {
		ChannelArr = input;
		this.bias = bias;
	}

	public override int InputLength => ChannelArr.Length;

	public override double GetWeightValue(int index) =>
		index == InputLength ? GetBiasValue() : ChannelArr[index].Weight;

	public override IInputDescriptor DescribeInput(int index) => ChannelArr[index].InputDescriptor;

	public override double GetBiasValue() => bias;

	public double SetBias(double value) => bias = value;

	public virtual int BiasIndex => InputLength;

	public void AddInputChannel(SignalChannel[] addInput) {
		SignalChannel[] newArr = new SignalChannel[ChannelArr.Length + addInput.Length];
		for (int i = 0; i < ChannelArr.Length; i++) newArr[i] = ChannelArr[i];
		for (int i = 0; i < addInput.Length; i++) newArr[ChannelArr.Length + i] = addInput[i];
		ChannelArr = newArr;
	}

	public void RemoveInputChannel(int iIndex, byte range = 1) {
		SignalChannel[] newArr = new SignalChannel[ChannelArr.Length - range];
		for (int i = 0; i < iIndex; i++) newArr[i] = ChannelArr[i];
		for (int i = iIndex + range; i < ChannelArr.Length; i++) newArr[i - range] = ChannelArr[i];
		ChannelArr = newArr;
	}

	#region subclasses
	public sealed class CustomActivated(IInputDescriptor[] input, IActivation activation, Random? weightStarter = null) : OutputNeuron(input, weightStarter) {
		public override IActivation GetActivation() => activation;
	}

	public sealed class ReLUActivated(IInputDescriptor[] input, Random? weightStarter = null) : OutputNeuron(input, weightStarter) {
		public override IActivation GetActivation() => Function.ReLU.Shared;
	}

	public sealed class SigmoidActivated(IInputDescriptor[] input, Random? weightStarter = null) : OutputNeuron(input, weightStarter) {
		public override IActivation GetActivation() => Function.Sigmoid.Shared;
	}

	public sealed class TanhActivated(IInputDescriptor[] input, Random? weightStarter = null) : OutputNeuron(input, weightStarter) {
		public override IActivation GetActivation() => Function.Tanh.Shared;
	}
	#endregion

}
