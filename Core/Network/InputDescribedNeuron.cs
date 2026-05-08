using System.Text;

namespace EliasFM.ArtificialNeuralNet.Core.Network;


public abstract class InputDescribedNeuron : Neuron {

	public enum InputSourceType { NetworkInput, HiddenNeuron, AuxInput }
	public interface IInputDescriptor {
		InputSourceType SourceType { get; }
		int IndexInSource { get; }
		string ToString();
	}
	public abstract IInputDescriptor DescribeInput(int inputIndex);

	public override string ToString() {
		int inputLength = InputLength;
		StringBuilder sb = new(GetType().Name + '{' + $"bias:{GetBiasValue()},inputs[{inputLength}]:[");
		for (int i = 0; i < inputLength; i++)
			sb.Append($"#{i}:" + '{' + $"ref:{DescribeInput(i)},weight:{GetWeightValue(i)}" + "},");
		return sb.Remove(sb.Length - 1, 1).Append("]}").ToString();
	}

	public abstract class HiddenNeuron(double precomputedOutput = 0) : InputDescribedNeuron {
		public double ComputeOutput(double[] input, bool storeOutput) {
			return storeOutput ? (StoredOutput = ComputeOutput(input)) : ComputeOutput(input);
		}
		public double StoredOutput { get; private set; } = precomputedOutput;

		public override string ToString() {
			string baseStr = base.ToString(), addStr = $", StoredOutput:{StoredOutput}";
			StringBuilder sb = new StringBuilder(baseStr.Length + addStr.Length).Append(baseStr);
			return sb.Insert(sb.Length - 1, addStr).ToString();
		}
	}
}




