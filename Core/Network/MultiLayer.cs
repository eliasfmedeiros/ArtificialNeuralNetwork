using System.Collections;
using System.Text;

namespace EliasFM.ArtificialNeuralNet.Core.Network;

public abstract class MultiLayer : INetwork, IEnumerable<INetwork> {
	public abstract INetwork LayerAt(int index);
	public abstract int LayerCount { get; }

	public double[] Predict(double[] input) {
		for (int i = 0; i < LayerCount; i++)
			input = LayerAt(i).Predict(input);
		return input;
	}

	public int InputLength => LayerAt(0).InputLength;

	public int OutputLength => LayerAt(LayerCount - 1).OutputLength;

	public IEnumerator<INetwork> GetEnumerator() {
		for (int i = 0; i < LayerCount; i++)
			yield return LayerAt(i);
	}

	IEnumerator IEnumerable.GetEnumerator() {
		return GetEnumerator();
	}

	public override string ToString() {
		int layers = LayerCount;
		StringBuilder sb = new($"{GetType().Name}" + '{' + $"Layers[{layers}]:[");
		for (int i = 0; i < layers; i++) sb.Append($"#{i}:{LayerAt(i)},");
		return sb.Remove(sb.Length - 1, 1).Append("]}").ToString();
	}
}
