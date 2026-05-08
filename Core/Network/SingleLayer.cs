using System.Collections;
using System.Text;

namespace EliasFM.ArtificialNeuralNet.Core.Network;

public abstract class SingleLayer : INetwork, IEnumerable<Neuron> {
	public abstract Neuron NeuronAt(int index);
	public abstract int NeuronCount { get; }
	public abstract int InputLength { get; }

	public double[] Predict(double[] input) {
		double[] output = new double[NeuronCount];
		for (int i = 0; i < output.Length; i++)
			output[i] = NeuronAt(i).ComputeOutput(input);
		return output;
	}

	public double[][] ToMatrix() {
		double[][] matrix = new double[NeuronCount][];
		for (int i = 0; i < matrix.Length; i++)
			matrix[i] = NeuronAt(i).ToArray();
		return matrix;
	}


	public virtual int OutputLength { get => NeuronCount; }

	public IEnumerator<Neuron> GetEnumerator() {
		for (int i = 0; i < NeuronCount; i++)
			yield return NeuronAt(i);
	}

	IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

	public override string ToString() {
		int neurons = NeuronCount;
		StringBuilder sb = new($"{GetType().Name}" + '{' + $"Neurons[{neurons}]:[");
		for (int i = 0; i < neurons; i++) sb.Append($"#{i}:{NeuronAt(i)},");
		return sb.Remove(sb.Length - 1, 1).Append("]}").ToString();
	}
}