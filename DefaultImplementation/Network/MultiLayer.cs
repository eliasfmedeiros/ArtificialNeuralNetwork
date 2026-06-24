
using EliasFM.ArtificialNeuralNet.Core;
using EliasFM.ArtificialNeuralNet.Core.Network;
using MultiLayerBase = EliasFM.ArtificialNeuralNet.Core.Network.MultiLayer;

namespace EliasFM.ArtificialNeuralNet.CommonImplementation.Network;

public class MultiLayer : MultiLayerBase {
	private readonly INetwork[] layerArr;

	public MultiLayer(INetwork[] layerArr) {
		this.layerArr = layerArr;
	}
	public MultiLayer(int input, OutputDescriptor[] descriptorArr) {
		layerArr = new INetwork[descriptorArr.Length];
		for (int i = 0; i < descriptorArr.Length; i++) {
			layerArr[i] = new SingleLayer(input, descriptorArr[i].ActivatorArr);
			input = layerArr[i].OutputLength;
		}
	}

	public override int LayerCount => layerArr.Length;

	public override INetwork LayerAt(int index) {
		return layerArr[index];
	}

	public readonly struct OutputDescriptor {
		public readonly IActivation[] ActivatorArr;
	}
}
