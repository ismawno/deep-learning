package deepLearning.utilities;

import java.io.Serializable;

import tensors.Float.*;

public class Dense implements Serializable {
	
	private static final long serialVersionUID = 1L;
		
	private int neurons;
	private int inputShape;
	
	private Matrix weights;
	private Vector bias, outVal, outDer;
	
	private Sequential.ACTIVATION actLabel;
	private ActivationFunction activation;
	
	private boolean ready;
	
	public Dense(int inputShape, int neurons, Sequential.ACTIVATION actLabel) {
		
		this(neurons, actLabel);
		init(inputShape);
	}
	
	public Dense(int neurons, Sequential.ACTIVATION actLabel) {
		
		this.neurons = neurons;
		
		this.actLabel = actLabel;
		activation = new ActivationFunction(actLabel);
		
		ready = false;
	}
	
	public Dense(Dense copied) {
		
		neurons = copied.neurons;
		actLabel = copied.actLabel;
		
		activation = new ActivationFunction(actLabel);
		ready = copied.isReady();
		
		if (ready) {
			
			weights = copied.weights.copy();
			bias = copied.bias.copy();
			
			inputShape = copied.getInputShape();
		}
	}
	
	public Dense init(int inputShape) {
		
		outVal = new Vector(getNeurons());
		outDer = new Vector(getNeurons());
		
		bias = new Vector(getNeurons());
		bias.randomize(1.0f);
		
		weights = new Matrix(getNeurons(), inputShape);
		weights.randomize(1.0f);
		
		if (getLabel() == Sequential.ACTIVATION.RELU)
			weights.mult((float) Math.sqrt(2.0f / inputShape));
		else
			weights.mult(1.0f / (float) Math.sqrt(inputShape));
		
		this.inputShape = inputShape;
		
		getReady();
		return this;
	}
	
	public Vector computeOutput(Vector inputs) {
		
		Vector raw = Vector.mult(weights, inputs);
		raw.add(bias);
		getActivation().compute(raw);
		
		outVal = getActivation().getValue();
		outDer = getActivation().getDerivative();
		
		return outVal;
	}
	
	public Dense randomize(float deviation) {
		
		getWeights().randomize(deviation);
		getBias().randomize(deviation);
		return this;
	}
	
	public Dense copy() {return new Dense(this);}
	public Matrix getWeights() {return weights;}
	public Vector getBias() {return bias;}
	public Vector getValue() {return getActivation().getValue();}
	public Vector getDerivative() {return getActivation().getDerivative();}
	public Vector getOutVal() {return outVal;}
	public Vector getOutDer() {return outDer;}
	public int getNeurons() {return neurons;}
	public int getInputShape() {return inputShape;}
	public Sequential.ACTIVATION getLabel() { return actLabel;}
	public ActivationFunction getActivation() {return activation;}
	public boolean isReady() {return ready;}
	public Dense getReady() {ready = true; return this;}
}
