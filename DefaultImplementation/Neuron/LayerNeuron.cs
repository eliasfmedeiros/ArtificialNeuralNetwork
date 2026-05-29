using EliasFM.ArtificialNeuralNet.Core;

namespace EliasFM.ArtificialNeuralNet.CommonImplementation.Neuron;

public abstract class LayerNeuron : Core.Neuron {

	public double[] WeightArr { get; }

	public LayerNeuron(int inputs) {
		WeightArr = new double[inputs + 1];
		for (int i = 0; i < WeightArr.Length; i++) WeightArr[i] = DefaultStartingWeight();
	}

	public override int InputLength => WeightArr.Length - 1;
	public override double GetWeightValue(int index) => WeightArr[index];

	public override double GetBiasValue() => WeightArr[BiasIndex];
	public virtual int BiasIndex => InputLength;

	#region subclasses
	public sealed class CustomActivation(int inputs, IActivation activation) : LayerNeuron(inputs) {
		public override IActivation GetActivation() => activation;
	}

	public sealed class ReLUActivation(int inputs) : LayerNeuron(inputs) {
		public override IActivation GetActivation() => Function.ReLU.Shared;
	}

	public sealed class SigmoidActivation(int inputs) : LayerNeuron(inputs) {
		public override IActivation GetActivation() => Function.Sigmoid.Shared;
	}

	public sealed class TanhActivation(int inputs) : LayerNeuron(inputs) {
		public override IActivation GetActivation() => Function.Tanh.Shared;
	}
	#endregion
}
