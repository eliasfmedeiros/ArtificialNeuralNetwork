
using EliasFM.ArtificialNeuralNet.Core.Network;
using MultiLayerBase = EliasFM.ArtificialNeuralNet.Core.Network.MultiLayer;

namespace EliasFM.ArtificialNeuralNet.CommonImplementation.Network;

public class MultiLayer : MultiLayerBase {
	private readonly INetwork[] layerArr;

	public MultiLayer(INetwork[] layerArr) {
		this.layerArr = layerArr;
	}

	public override int LayerCount => layerArr.Length;

	public override INetwork LayerAt(int index) {
		return layerArr[index];
	}
}
