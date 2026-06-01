using System.Globalization;
using System.Numerics;
using System.Security.AccessControl;
using static EliasFM.ArtificialNeuralNet.Core.Network.FreeNN;

namespace EliasFM.ArtificialNeuralNet.CommonImplementation.Neuron.Free.Auxiliary;

public struct InputDescriptor(InputSourceType sourceType, int indexInSource) : InputDescribedNeuron.IInputDescriptor {
	#region constants and static methods
	public const int AuxInputBase = int.MinValue / 2;
	public const int NetworkInputBase = int.MinValue;
	public const int ExternalInputCardinality = -AuxInputBase;

	public static int Encode(InputSourceType sourceType, int indexInSource) {
		ArgumentOutOfRangeException.ThrowIfNegative(indexInSource);
		if (sourceType == InputSourceType.HiddenNeuron) return indexInSource;
		ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(indexInSource, ExternalInputCardinality);
		if (sourceType == InputSourceType.NetworkInput) return NetworkInputBase + indexInSource;
		return AuxInputBase + indexInSource;
	}

	public static InputSourceType DecodeType(int code) {
		if (code < AuxInputBase) return InputSourceType.NetworkInput;
		if (code < 0) return InputSourceType.AuxInput;
		return InputSourceType.HiddenNeuron;
	}
	public static int DecodeIndex(int code) {
		if (code < AuxInputBase) return (int)(code + 2L * ExternalInputCardinality);
		if (code < 0) return code + ExternalInputCardinality;
		return code;
	}
	#endregion

	public int Code { get; private set; } = Encode(sourceType, indexInSource);

	public InputSourceType SourceType {
		readonly get => DecodeType(Code);
		set => Code = Encode(value, IndexInSource);
	}

	public int IndexInSource {
		readonly get => DecodeIndex(Code);
		set => Code = Encode(SourceType, value);
	}

	public override readonly string ToString() {
		return GetType().Name + "{" + $"SourceType:{SourceType},IndexInSource:{IndexInSource}" + "}";
	}

}



