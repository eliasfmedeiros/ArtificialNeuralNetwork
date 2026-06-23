using System.Text;

namespace EliasFM.ArtificialNeuralNet.Core.Network;

public abstract partial class FreeNN : INetwork {

	public long TotalNeuronCount => HiddenNeuronCount + OutputLength;

	public abstract int HiddenNeuronCount { get; }

	public abstract HiddenNeuron HiddenNeuronAt(int index);

	public abstract InputDescribedNeuron OutputNeuronAt(int index);

	private double[] BuildNeuronInput(InputDescribedNeuron neuron, double[] networkInput) { // neuronios são sempre acessíveis, mas networkInput só existe durante chamada de método
		double[] neuronInput = new double[neuron.InputLength];
		for (int iInput = 0; iInput < neuronInput.Length; iInput++) {
			InputDescribedNeuron.IInputComponentIdentity inputDescription = neuron.IdentifyInputComponent(iInput);
			switch (inputDescription.SourceType) {
				case InputSourceType.HiddenNeuron: neuronInput[iInput] = HiddenNeuronAt(inputDescription.IndexInSource).StoredOutput; break;
				case InputSourceType.NetworkInput: neuronInput[iInput] = networkInput[inputDescription.IndexInSource]; break;
				default: neuronInput[iInput] = AuxInputAt(inputDescription.IndexInSource); break;
			}
		}
		return neuronInput;
	}

	public virtual double AuxInputAt(int indexInSource) => throw new NotImplementedException($"{this}.AuxInputAt({+indexInSource})");

	public double[] Predict(double[] networkInput) {
		int hiddenNeurons = HiddenNeuronCount;
		for (int iNeuron = 0; iNeuron < hiddenNeurons; iNeuron++) {
			HiddenNeuron neuron = HiddenNeuronAt(iNeuron);
			neuron.ComputeOutput(BuildNeuronInput(neuron, networkInput), true);
		}
		double[] output = new double[OutputLength];
		for (int iNeuron = 0; iNeuron < output.Length; iNeuron++) {
			InputDescribedNeuron neuron = OutputNeuronAt(iNeuron);
			output[iNeuron] = neuron.ComputeOutput(BuildNeuronInput(neuron, networkInput));
		}
		return output;
	}

	public abstract int InputLength { get; }

	public abstract int OutputLength { get; }

	public override string ToString() {
		int hiddens = HiddenNeuronCount, outputs = OutputLength;
		StringBuilder sb = new(GetType().Name + '{' +
			$"inputs:{InputLength},HiddenNeurons[{hiddens}]:[");
		for (int i = 0; i < hiddens; i++) sb.Append($"#{i}:{HiddenNeuronAt(i)},");
		sb.Remove(sb.Length - 1, 1).Append($"],OutputNeurons[{outputs}]:[");
		for (int i = 0; i < outputs; i++) sb.Append($"#{i}:{OutputNeuronAt(i)},");
		return sb.Remove(sb.Length - 1, 1).Append("]}").ToString();
	}

	#region Subclasses
	public enum InputSourceType { NetworkInput, HiddenNeuron, AuxInput }

	public abstract class InputDescribedNeuron : Neuron {
		public interface IInputComponentIdentity {
			InputSourceType SourceType { get; }
			int IndexInSource { get; }
			string ToString();
		}
		public abstract IInputComponentIdentity IdentifyInputComponent(int index);

		public override string ToString() {
			int inputLength = InputLength;
			StringBuilder sb = new(GetType().Name + '{' + $"bias:{GetBiasValue()},inputs[{inputLength}]:[");
			for (int i = 0; i < inputLength; i++)
				sb.Append($"#{i}:" + '{' + $"ref:{IdentifyInputComponent(i)},weight:{GetWeightValue(i)}" + "},");
			return sb.Remove(sb.Length - 1, 1).Append("]}").ToString();
		}
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

	#endregion

}


