package deepLearning.utilities;

import java.io.Serializable;

import tensors.Float.*;

public class Filter implements Serializable {
	
	private static final long serialVersionUID = 1L;

	private Matrix[] layers;
	private int[] shape;
		
	public Filter(int[] shape) {
		
		if (shape.length != 3)
			throw new RuntimeException("Shape dimensions must be 3 in Filter constructor");
		
		layers = new Matrix[shape[2]];
		for (int i = 0; i < layers.length; i++) {
			layers[i] = new Matrix(shape[0], shape[1]);
			layers[i].randomize(1.0f);
		}
		
		this.shape = shape;
	}
	
	public Filter(Filter copied) {
		shape = copied.getShape();
		
		layers = new Matrix[shape[2]];
		for (int i = 0; i < layers.length; i++)
			layers[i] = copied.getLayers()[i].copy();
	}
	
	public Matrix convolve(Matrix[] input) {
		
		if (input.length != getShape()[2])
			throw new RuntimeException("Trying to convolve two volumes of different depth");
		
		Matrix[] convolvedVol = new Matrix[getShape()[2]];
		for (int i = 0; i < getShape()[2]; i++)
			convolvedVol[i] = Matrix.convolve(input[i], layers[i], 1);
		
		Matrix result = convolvedVol[0];
		for (int i = 1; i < getShape()[2]; i++)
			result.add(convolvedVol[i]);
		
		return result;
	}
	
	public Filter fix(Matrix[] addit) {
		
		if (addit.length != getLayers().length)
			throw new RuntimeException("Matrix addition mismatch in filter fix()");
		
		for (int i = 0; i < addit.length; i++)
			getLayers()[i].sub(addit[i]);
		return this;
	}
	
	public Filter randomize(float deviation) {for (Matrix mat : layers) mat.randomize(deviation); return this;}
	public Filter copy() {return new Filter(this);}
	public int[] getShape(){return shape;}
	public Matrix[] getLayers() {return layers;}
}
