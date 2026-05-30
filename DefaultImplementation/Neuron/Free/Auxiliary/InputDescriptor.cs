using System.Globalization;
using System.Numerics;
using System.Security.AccessControl;
using static EliasFM.ArtificialNeuralNet.Core.Network.FreeNN;

namespace EliasFM.ArtificialNeuralNet.CommonImplementation.Neuron.Free.Auxiliary;

public struct InputDescriptor(InputSourceType sourceType, int indexInSource) : InputDescribedNeuron.IInputDescriptor {
	public InputSourceType SourceType { get; set; } = sourceType;

	public int IndexInSource { get; set; } = indexInSource;

	public override readonly string ToString() {
		return GetType().Name + "{" + $"SourceType:{SourceType},IndexInSource:{IndexInSource}" + "}";
	}

}



