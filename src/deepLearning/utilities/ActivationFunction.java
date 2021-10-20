package deepLearning.utilities;

import java.io.Serializable;

import tensors.Float.*;

public class ActivationFunction implements Serializable {
	
	private static final long serialVersionUID = 1L;

	public Sequential.ACTIVATION actLabel;
	
	private Vector value;
	private Vector derivative;
	
	public ActivationFunction(Sequential.ACTIVATION actLabel_) {
		
		actLabel = actLabel_; 
	}
	
	public ActivationFunction compute(Vector raw) {
		
		value = raw.copy();
		derivative = raw.copy();
		
		if (actLabel == Sequential.ACTIVATION.SIGMOID)
			for (int i = 0; i < raw.getLength(); i++) {
				
				value.set(i, 1.0f / (1.0f + (float) Math.exp( - raw.get(i))));
				derivative.set(i, value.get(i) * (1.0f - value.get(i)));
			}
		else if (actLabel == Sequential.ACTIVATION.RELU)
			for (int i = 0; i < raw.getLength(); i++) {
				
				value.set(i, Math.max(0.0f, raw.get(i)));
				if (raw.get(i) > 0)
					derivative.set(i, 1.0f);
				else
					derivative.set(i, 0.0f);
			}
		else if (actLabel == Sequential.ACTIVATION.TANH)
			for (int i = 0; i < raw.getLength(); i++) {
				
				value.set(i, (float) Math.tanh(raw.get(i)));
				derivative.set(i, 1.0f - value.get(i) * value.get(i));
			}
		else if (actLabel == Sequential.ACTIVATION.SOFTPLUS)
			for (int i = 0; i < raw.getLength(); i++) {
				
				value.set(i, (float) Math.log(1.0f + Math.exp(raw.get(i))));
				derivative.set(i, 1.0f / (1.0f + (float) Math.exp( - raw.get(i))));
			}
		else if (actLabel == Sequential.ACTIVATION.BSTEP)
			for (int i = 0; i < raw.getLength(); i++) {
				
				if (raw.get(i) > 0)
					value.set(i, 1.0f);
				else
					value.set(i, 0.0f);
				
				derivative.set(i, 0.0f);
			}
		else if (actLabel == Sequential.ACTIVATION.SOFTMAX) {
			
			float expSum = 0;
			Vector subRaw = Vector.sub(raw, raw.max());
			for (int i = 0; i < raw.getLength(); i++)
				expSum += Math.exp(subRaw.get(i));
			
			for (int i = 0; i < subRaw.getLength(); i++) {
				
				value.set(i, (float) Math.exp(subRaw.get(i)) / expSum);
				derivative.set(i, value.get(i) * (1.0f - value.get(i)));
			}
		} else if (actLabel == Sequential.ACTIVATION.LINEAR)
			derivative.setAll(1);
		
		return this;
	}

	public Vector getValue() {return value;}
	public Vector getDerivative() {return derivative;}
	public Matrix getValueReshaped(int rows, int cols) {return new Matrix(value.get(), rows, cols);}
	public Matrix getDerivativeReshaped(int rows, int cols) {return new Matrix(derivative.get(), rows, cols);}
}
