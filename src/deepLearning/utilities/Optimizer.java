package deepLearning.utilities;

import java.io.Serializable;
import java.util.List;

import tensors.Float.*;

public class Optimizer implements Serializable {
	
	private static final long serialVersionUID = 1L;
	
	public enum LOSS {
		MSE,
		CROSSENTROPY;
	}

	private LOSS lossFunc;
	private Sequential model;
	
	private Matrix[] deltaW;
	private Matrix[][][] deltaF;
	
	private Vector[] deltaB;
	
	public Optimizer(Sequential model, LOSS lossFunc) {
		
		this.model = model;
		this.lossFunc = lossFunc;
		
		deltaW = new Matrix[model.getDenseCount()];
		deltaB = new Vector[model.getDenseCount()];
		
		for (int i = 0; i < model.getDenseCount(); i++) {
			Dense dense = model.getDense().get(i);
			
			deltaW[i] = new Matrix(dense.getWeights().getRows(), dense.getWeights().getCols());
			deltaB[i] = new Vector(dense.getBias().getLength());
		}
		
		deltaF = new Matrix[model.getConvCount()][][];
		
		for (int i = 0; i < model.getConvCount(); i++) {
			Conv2D conv = model.getConv().get(i);
			deltaF[i] = new Matrix[conv.getFilterCount()][conv.getFilterShape()[2]];
			
			for (int j = 0; j < deltaF[i].length; j++)
				for (int k = 0; k < deltaF[i][j].length; k++)
					deltaF[i][j][k] = new Matrix(conv.getFilterShape()[0], model.getConv().get(i).getFilterShape()[1]);
			
		}	
	}
	
	public void zeroGrad() {
		
		for (int i = 0; i < model.getDenseCount(); i++) {
			deltaW[i].setAll(0.0f);
			deltaB[i].setAll(0.0f);
		}
		
		for (int i = 0; i < model.getConvCount(); i++)
			for (int j = 0; j < deltaF[i].length; j++)
				for (int k = 0; k < deltaF[i][j].length; k++)
					deltaF[i][j][k].setAll(0.0f);
	}
	
	public float computeLoss(Vector guess, Vector label) {
			if (getLoss() == LOSS.MSE)
				return Vector.sub(guess, label).normSq();
			else if (label.max() == 1.0f)
				return - (float) Math.log(Math.max(guess.get(label.indexMax()), 1e-7));
			else {
				float sum = 0;
				for (int i = 0; i < guess.getLength(); i++)
					sum -= (float) label.get(i) * Math.log(Math.max(guess.get(i), 1e-7));
				return sum;
			}
	}
	
	public Vector computeLossDeriv(Vector guess, Vector label) {
		
		if (!model.hasDense())
			if (getLoss() == LOSS.MSE)
				return Vector.sub(guess, label);
			else
				return Vector.mult(Vector.divElementWise(label, guess), - 1.0f);
		else
			if (getLoss() == LOSS.MSE) {
				Vector vec = Vector.sub(guess, label);
				vec.multElementWise(model.getLastDense().getDerivative());
				return vec;
			} else
				return Vector.sub(guess, label);
	}
	
	public void fix(Sequential.Options options) {
		
		for (int i = 0; i < model.getDenseCount(); i++) {
			Dense dense = model.getDense().get(i);
			
			if (options.isRegularized())
				deltaW[i].add(Matrix.mult(deltaW[i], options.getRegFactor()));
			
			deltaW[i].mult(options.lr);
			deltaB[i].mult(options.lr);
			
			dense.getWeights().sub(deltaW[i]);
			dense.getBias().sub(deltaB[i]);
		}
		
		for (int i = 0; i < model.getConvCount(); i++) {
			Conv2D conv = model.getConv().get(i);
			
			for (int j = 0; j < conv.getFilterCount(); j++)
				for (int k = 0; k < conv.getFilterShape()[2]; k++) {
					
					if (options.isRegularized())
						deltaF[i][j][k].add(Matrix.mult(deltaF[i][j][k], options.getRegFactor()));
					
					deltaF[i][j][k].mult(options.lr);
					conv.getFilters()[j].getLayers()[k].sub(deltaF[i][j][k]);
				}
		}
	}
	
	public void backpropagate(Vector input, Vector lossDeriv) {
		
		List<Dense> denses = model.getDense();
		List<Conv2D> convs = model.getConv();
		
		Vector[] errDense = new Vector[denses.size()];
		Matrix[][] errConv = new Matrix[convs.size()][];
		for (int i = 0; i < convs.size(); i++)
			errConv[i] = new Matrix[convs.get(i).getFilterCount()];
		
		if (model.hasDense())
			errDense[denses.size() - 1] = lossDeriv;
		
		for (int i = denses.size() - 2; i >= 0; i--) {
			Dense fwd = denses.get(i + 1);
			Dense current = denses.get(i);
			
			errDense[i] = Vector.mult(Matrix.transpose(fwd.getWeights()), errDense[i + 1]);
			errDense[i].multElementWise(current.getDerivative());
		}
		
		if (model.hasConv()) {
			Conv2D last = model.getLastConv();
			
			if (model.hasDense()) {
				Dense first = model.getFirstDense();
				errConv[convs.size() - 1] = last.toErrorVolume(Vector.mult(Matrix.transpose(first.getWeights()), errDense[0]), model.getOrientation());
			} else
				errConv[convs.size() - 1] = last.toErrorVolume(lossDeriv, model.getOrientation());
			
			if (last.getPool() != Conv2D.POOLING.NONE)
				errConv[convs.size() - 1] = unPool(errConv[convs.size() - 1], last);
			
			for (int i = 0; i < model.getLastConv().getFilterCount(); i++)
				errConv[convs.size() - 1][i].multElementWise(last.getOutDer()[i]);
		}
		
		for (int i = convs.size() - 2; i >= 0; i--) {
			Conv2D fwd = convs.get(i + 1);
			Conv2D current = convs.get(i);
			
			for (int j = 0; j < current.getFilterCount(); j++) {
				errConv[i][j] = new Matrix(current.getPooledOutShape()[0], current.getPooledOutShape()[1]);
				for (int k = 0; k < fwd.getFilterCount(); k++) //Aqui, si se usa padding, hay que UNPADDEAR: .add(unPad(lo que sea))
					if (fwd.getPad() == Conv2D.PADDING.VALID)
						errConv[i][j].add(Matrix.convolve(Matrix.zeroPad(errConv[i + 1][k], fwd.getFilterShape()[0] - 1, fwd.getFilterShape()[1] - 1), Matrix.rotate(fwd.getFilters()[k].getLayers()[j], - 2), 1));
					else
						errConv[i][j].add(Matrix.unPad(Matrix.convolve(Matrix.zeroPad(errConv[i + 1][k], fwd.getFilterShape()[0] - 1, fwd.getFilterShape()[1] - 1), Matrix.rotate(fwd.getFilters()[k].getLayers()[j], - 2), 1), fwd.getPadLayers()[0], fwd.getPadLayers()[2]));
			}
			
			if (current.getPool() != Conv2D.POOLING.NONE)
				errConv[i] = unPool(errConv[i], current);
			
			for (int j = 0; j < current.getFilterCount(); j++)
				errConv[i][j].multElementWise(current.getOutDer()[j]);
		}
		
		for (int i = 0; i < convs.size(); i++) {
			
			Conv2D current = convs.get(i);
			if (i != 0) {
				Conv2D prev = convs.get(i - 1);
				
				for (int j = 0; j < current.getFilterCount(); j++)
					for (int k = 0; k < prev.getFilterCount(); k++)
						if (current.getPad() == Conv2D.PADDING.VALID)
							deltaF[i][j][k].add(Matrix.convolve(prev.getPooledOutVal()[k], errConv[i][j], 1));
						else
							deltaF[i][j][k].add(Matrix.convolve(Matrix.zeroPad(prev.getPooledOutVal()[k], current.getPadLayers()[0], current.getPadLayers()[1]), errConv[i][j], 1));
			} else
				for (int j = 0; j < current.getFilterCount(); j++)
					for (int k = 0; k < current.getInputShape()[2]; k++)
						if (current.getPad() == Conv2D.PADDING.VALID)
							deltaF[i][j][k].add(Matrix.convolve(current.toInputVolume(input, model.getOrientation())[k], errConv[i][j], 1));
						else
							deltaF[i][j][k].add(Matrix.convolve(Matrix.zeroPad(current.toInputVolume(input, model.getOrientation())[k], current.getPadLayers()[0], current.getPadLayers()[1]), errConv[i][j], 1));
		}
		
		for (int i = 0; i < denses.size(); i++) {
			
			if (i != 0) {
				Dense prev = denses.get(i - 1);
				deltaW[i].add(errDense[i].mult(prev.getValue()));
			} else if (!model.hasConv())
				deltaW[i].add(errDense[i].mult(input));
			else
				deltaW[i].add(errDense[i].mult(model.getLastConv().getFlatOutput()));
			
			deltaB[i].add(errDense[i]);
		}
	}
	
	private Matrix[] unPool(Matrix[] pooledErrors, Conv2D conv) {
		
		Matrix[] unPooledErrors = new Matrix[pooledErrors.length];
		int stride = conv.getPoolStride();
		
		for (int i = 0; i < pooledErrors.length; i++) {
			
			Matrix values = conv.getOutVal()[i];
			Matrix pooled = pooledErrors[i];
			Matrix unPooled = new Matrix(values.getRows(), values.getCols());
			
			int x;
			int y;
			
			for (int j = 0; j < pooled.getRows(); j++) {
				x = j * stride;
				for (int k = 0; k < pooled.getCols(); k++) {
					y = k * stride;
					
					Matrix portionValue = values.getPortion(x, y, stride, stride);
					Matrix portionUnPooled = unPooled.getPortion(x, y, stride, stride);
					
					if (conv.getPool() == Conv2D.POOLING.MAX) {
						
						int[] maxIndex = portionValue.indexMax();
						portionUnPooled.set(maxIndex[0], maxIndex[1], pooled.get(j, k));
					} else if (conv.getPool() == Conv2D.POOLING.AVG)
						portionUnPooled.setAll(pooled.get(j, k) / (stride * stride));
					
					unPooled.setPortion(portionUnPooled, x, y);
				}
			}
			
			unPooledErrors[i] = unPooled;
		}
		
		return unPooledErrors;
	}
	
	public LOSS getLoss() {return lossFunc;}
	public Sequential getModel() {return model;}
	public Matrix[] getDeltaW() {return deltaW;}
	public Matrix[][][] getDeltaF() {return deltaF;}
	public Vector[] getDeltaB() {return deltaB;}
}
