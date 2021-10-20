package deepLearning.utilities;

import java.io.Serializable;

import tensors.Float.*;

public class Conv2D implements Serializable {
	
	private static final long serialVersionUID = 1L;

	public enum POOLING {
		MAX,
		AVG,
		NONE;
	}
	
	public enum PADDING {
		ORIGINAL,
		VALID;
	}
	
	private PADDING pad;
	private POOLING pool;
	
	private Sequential.ACTIVATION actLabel;
	private ActivationFunction activation;
	
	private Filter[] filters;
	
	private int filterCount, poolStride;
	private int[] filterShape, inputShape, outputShape, pooledOutShape, padLayers;
	
	private Matrix[] outVal, outDer, pooledOutVal;
	private Vector flatOutput;
	
	private boolean ready;
	
	public Conv2D(int filterCount, int[] filterShapeRed, Sequential.ACTIVATION actLabel) {
		
		if (filterShapeRed.length != 2)
			throw new RuntimeException("Filter shape dimensions must be 2 in Conv2D constructor");
		
		this.filterCount = filterCount;
		this.actLabel = actLabel;
		
		filterShape = new int[3];
		
		filterShape[0] = filterShapeRed[0];
		filterShape[1] = filterShapeRed[1];
		
		activation = new ActivationFunction(actLabel);
		
		poolStride = 1;
		
		pool = POOLING.NONE;
		pad = PADDING.VALID;
		
		outVal = new Matrix[filterCount];
		outDer = new Matrix[filterCount];
		pooledOutVal = new Matrix[filterCount];
		
		ready = false;
	}
	
	public Conv2D(int[] inputShape, int filterCount, int[] filterShapeRed, Sequential.ACTIVATION actLabel) {
		
		this(filterCount, filterShapeRed, actLabel);
		init(inputShape);
	}
	
	public Conv2D(Conv2D copied) {
		
		filterCount = copied.getFilterCount();
		filterShape = copied.getFilterShape();
		actLabel = copied.getLabel();
		
		activation = new ActivationFunction(actLabel);
		
		poolStride = copied.getPoolStride();
		
		pool = copied.getPool();
		pad = copied.getPad();
		
		outVal = new Matrix[filterCount];
		outDer = new Matrix[filterCount];
		pooledOutVal = new Matrix[filterCount];
		
		ready = copied.isReady();
		
		if (ready) {
			
			inputShape = copied.getInputShape();
			outputShape = copied.getOutputShape();
			
			pooledOutShape = copied.getPooledOutShape();
			padLayers = copied.getPadLayers();
			
			filters = new Filter[filterCount];
			for (int i = 0; i < filterCount; i++)
				filters[i] = copied.getFilters()[i].copy();
		}
	}
	
	public Conv2D init(int[] inputShape) {
		
		if (inputShape.length != 3)
			throw new RuntimeException("Input shape dimensions must be 3 in Conv2D constructor");
		
		filterShape[2] = inputShape[2];
		
		filters = new Filter[filterCount];
		for (int i = 0; i < filters.length; i++)
			filters[i] = new Filter(filterShape);
		
		this.inputShape = inputShape;
		
		if (getPad() == PADDING.VALID) 
			outputShape = Matrix.afterConv(inputShape, filterShape, 1);
			
		else if (getPad() == PADDING.ORIGINAL) {
			padLayers = Matrix.toMaintainConvDim(inputShape, filterShape, 1);
			outputShape = Matrix.afterConv(new int[] {inputShape[0] + 2 * padLayers[0], inputShape[1] + 2 * padLayers[1], 0}, filterShape, 1);
		}
		
		outputShape[2] = filterCount;
		
		pooledOutShape = Matrix.afterPool(outputShape, poolStride);
		pooledOutShape[2] = filterCount;
		
		getReady();
		return this;
	}
	
	public Matrix[] computeOutput(Matrix[] unpaddedInput) {
		
		Matrix[] input;
		
		if (getPad() == PADDING.ORIGINAL) {
			input = new Matrix[unpaddedInput.length];
			
			for (int i = 0; i < unpaddedInput.length; i++)
				input[i] = Matrix.zeroPad(unpaddedInput[i], getPadLayers()[0], getPadLayers()[1]);
		} else
			input = unpaddedInput;
		
		Matrix[] raw = new Matrix[getFilterCount()];
		for (int i = 0; i < getFilterCount(); i++) {
			
			raw[i] = getFilters()[i].convolve(input);			
			getActivation().compute(raw[i].flatten());
			
			outVal[i] = getActivation().getValueReshaped(getOutputShape()[0], getOutputShape()[1]);
			outDer[i] = getActivation().getDerivativeReshaped(getOutputShape()[0], getOutputShape()[1]);
			
			if (getPool() == POOLING.MAX)
				pooledOutVal[i] = Matrix.maxPooling(outVal[i], poolStride);
			else if (getPool() == POOLING.AVG)
				pooledOutVal[i] = Matrix.averagePooling(outVal[i], poolStride);
			else
				pooledOutVal[i] = outVal[i];
		}
		
		return pooledOutVal;
	}
	
	public Matrix[] computeOutput(Vector input, String orientation) {
		
		return computeOutput(toInputVolume(input, orientation));
	}
	
	public Matrix[] toInputVolume(Vector input, String orientation) {
		
		if (orientation.equals("horizontal")) {
			Matrix undivided = new Matrix(input.get(), getInputShape()[0], getInputShape()[1] * getInputShape()[2]);
			return undivided.divideHor(getInputShape()[2]);
		} else {
			Matrix undivided = new Matrix(input.get(), getInputShape()[0] * getInputShape()[2], getInputShape()[1]);
			return undivided.divideVer(getInputShape()[2]);
		}
	}
	
	public Matrix[] toErrorVolume(Vector error, String orientation) {
		
		if (orientation.equals("horizontal")) {
			Matrix undivided = new Matrix(error.get(), getPooledOutShape()[0], getPooledOutShape()[1] * getPooledOutShape()[2]);
			return undivided.divideHor(getPooledOutShape()[2]);
		} else {
			Matrix undivided = new Matrix(error.get(), getPooledOutShape()[0] * getPooledOutShape()[2], getPooledOutShape()[1]);
			return undivided.divideVer(getPooledOutShape()[2]);
		}
	}
	
	public Conv2D randomize(float deviation) {for (Filter filter : filters) filter.randomize(deviation); return this;}
	public Conv2D getReady() {ready = true; return this;}
	public Vector getFlatOutput() {return flatOutput;}
	public Conv2D copy() {return new Conv2D(this);}
	public Filter[] getFilters() {return filters;}
	public Matrix[] getOutVal() {return outVal;}
	public Matrix[] getOutDer() {return outDer;}
	public Matrix[] getPooledOutVal() {return pooledOutVal;}
	public int getFilterCount() {return filterCount;}
	public int[] getFilterShape() {return filterShape;}
	public int[] getInputShape() {return inputShape;}
	public int[] getOutputShape() {return outputShape;}
	public int[] getPooledOutShape() {return pooledOutShape;}
	public int getInputSize() {return inputShape[0] * inputShape[1] * inputShape[2];}
	public int getPooledOutSize() {return pooledOutShape[0] * pooledOutShape[1] * pooledOutShape[2];}
	public int getFilterSize() {return filterShape[0] * filterShape[1] * filterShape[2];}
	public int getPoolStride() {return poolStride;}
	public ActivationFunction getActivation() {return activation;}
	public Sequential.ACTIVATION getLabel(){return actLabel;}
	public PADDING getPad() {return pad;}
	public int[] getPadLayers() {return padLayers;}
	public POOLING getPool() {return pool;}
	public boolean isReady() {return ready;}
	
	public Conv2D setPool(POOLING method, int stride) {pool = method; poolStride = stride; return this;}
	public Conv2D setPad(PADDING method) {pad = method; return this;}
	public Conv2D setFlat(Vector flat) {flatOutput = flat; return this;}
}
