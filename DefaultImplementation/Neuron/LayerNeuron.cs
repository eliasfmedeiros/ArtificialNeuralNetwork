using EliasFM.ArtificialNeuralNet.Core;

namespace EliasFM.ArtificialNeuralNet.CommonImplementation.Neuron;

public abstract class LayerNeuron : Core.Neuron {

	public double[] WeightArr { get; }

	public LayerNeuron(int inputs, Random? weightStarter = null) {
		WeightArr = new double[inputs + 1];
		for (int i = 0; i < WeightArr.Length; i++) WeightArr[i] = DefaultStartingWeight(weightStarter);
	}

	public override int InputLength => WeightArr.Length - 1;
	public override double GetWeightValue(int index) => WeightArr[index];

	public override double GetBiasValue() => WeightArr[BiasIndex];
	public virtual int BiasIndex => InputLength;

	#region subclasses
	public sealed class CustomActivated(int inputs, IActivation activation, Random? weightStarter = null) : LayerNeuron(inputs, weightStarter) {
		public override IActivation GetActivation() => activation;
	}

	public sealed class ReLUActivated(int inputs, Random? weightStarter = null) : LayerNeuron(inputs, weightStarter) {
		public override IActivation GetActivation() => Function.ReLU.Shared;
	}

	public sealed class SigmoidActivated(int inputs, Random? weightStarter = null) : LayerNeuron(inputs, weightStarter) {
		public override IActivation GetActivation() => Function.Sigmoid.Shared;
	}

	public sealed class TanhActivated(int inputs, Random? weightStarter = null) : LayerNeuron(inputs, weightStarter) {
		public override IActivation GetActivation() => Function.Tanh.Shared;
	}
	#endregion
}
