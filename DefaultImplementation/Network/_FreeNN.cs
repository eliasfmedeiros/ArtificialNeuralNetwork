namespace EliasFM.ArtificialNeuralNet.CommonImplementation.Network;

public class FreeNN(int i, Core.Network.FreeNN.HiddenNeuron[] h, Core.Network.FreeNN.InputDescribedNeuron[] o) : Core.Network.FreeNN
{
	private readonly int inputLengthCache = i;

	public HiddenNeuron[] HiddenArr { get; } = h;
	public InputDescribedNeuron[] OutputArr { get; } = o;

	public override int InputLength => inputLengthCache;
	public override int HiddenNeuronCount => HiddenArr.Length;
	public override int OutputLength => OutputArr.Length;

	public override HiddenNeuron HiddenNeuronAt(int index) => HiddenArr[index];

	public override InputDescribedNeuron OutputNeuronAt(int index) => OutputArr[index];

	public override string ToString()
	{
		return base.ToString();
	}

}
