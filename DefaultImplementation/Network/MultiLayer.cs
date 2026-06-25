using EliasFM.ArtificialNeuralNet.Core;
using EliasFM.ArtificialNeuralNet.Core.Network;
using SingleLayerBase = EliasFM.ArtificialNeuralNet.Core.Network.SingleLayer;
using MultiLayerBase = EliasFM.ArtificialNeuralNet.Core.Network.MultiLayer;

namespace EliasFM.ArtificialNeuralNet.CommonImplementation.Network;

public class MultiLayer : MultiLayerBase {
	private readonly INetwork[] layerArr;

	public MultiLayer(int input, OutputDescriptor[] descriptorArr) {
		layerArr = new INetwork[descriptorArr.Length];
		for (int i = 0; i < descriptorArr.Length; i++) {
			layerArr[i] = new SingleLayer(input, descriptorArr[i].ActivatorArr);
			input = layerArr[i].OutputLength;
		}
	}

	public override int LayerCount => layerArr.Length;

	public override INetwork LayerAt(int index) => layerArr[index];

	public SingleLayerBase SingleLayerAt(int index) => (SingleLayerBase)layerArr[index];

	public readonly struct OutputDescriptor {
		public IActivation[] ActivatorArr { get; }

		public OutputDescriptor(IActivation[] activators) => ActivatorArr = activators;

		public OutputDescriptor(IActivation activator, int length) {
			ActivatorArr = new IActivation[length];
			while (length-- > 0) ActivatorArr[length] = activator;
		}
	}
}
