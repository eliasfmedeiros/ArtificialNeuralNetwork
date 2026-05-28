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

}
