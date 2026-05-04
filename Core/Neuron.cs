using System.Text;

namespace EliasFM.ArtificialNeuralNet.Core;

public abstract class Neuron : INeural, IActivator {
	public abstract int InputLength { get; }
	public abstract double GetWeightValue(int index);
	public abstract double GetBiasValue();

	protected virtual double MultiplyWeight(double input, int weightIndex) => input * GetWeightValue(weightIndex);

	public virtual double WeightedSum(double[] input) {
		double sum = GetBiasValue();
		for (int i = 0; i < InputLength; i++)
			sum += MultiplyWeight(input[i], i);
		return sum;
	}

	public abstract double ApplyActivationFunction(double x);

	public double ComputeOutput(double[] input) {
		return ApplyActivationFunction(WeightedSum(input));
	}

	public int OutputLength => 1;

	public double[] ToArray() {
		double[] array = new double[InputLength + 1];
		for (int i = 0; i < InputLength; i++)
			array[i] = GetWeightValue(i);
		array[InputLength] = GetBiasValue();
		return array;
	}

	public override string ToString() {
		int inputLength = InputLength;
		StringBuilder sb = new(GetType().Name + '{' + $"InputWeights[{inputLength}]:[");
		for (int i = 0; i < inputLength; i++) sb.Append($"#{i}:{GetWeightValue(i)},");
		return sb.Remove(sb.Length - 1, 1).Append($"],bias:{GetBiasValue()}").Append('}').ToString();
	}

	private static Random NewRandom() => new(Random.Shared.Next() + Environment.TickCount);
	private static double BlockedRandom = NewRandom().NextDouble();

	public static double Random1Inclusive((Random main, Random antagonist)? generators = null) {
		Random mainGenerator = generators == null ? Random.Shared : generators.Value.main;
		byte t = 0;
		while (t++ < 2) {
			double candidate = mainGenerator.NextDouble();
			if (candidate != BlockedRandom) return candidate;
			BlockedRandom = (generators == null ? NewRandom() : generators.Value.antagonist).NextDouble();
			if (candidate != 0) return 1;
		}
		return mainGenerator.NextDouble() < 0.5d ? mainGenerator.NextDouble() : 1 - mainGenerator.NextDouble();
	}

	public static double DefaultStartingWeight(Random? rdn = null) => Random1Inclusive(rdn == null ? null : (rdn, rdn)) * 2 - 1;
	//public static double DefaultStartingWeight(Random? rdn = null) => Math.Round(Random1Inclusive(rdn==null?null:(rdn, rdn)) * 2 - 1, 2);

}
