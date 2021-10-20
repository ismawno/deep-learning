package deepLearning.utilities;

import processing.core.*;
import tensors.Float.*;

import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.io.File;
import java.io.IOException;
import java.io.FileNotFoundException;

public class Sequential extends PApplet implements Serializable {
	
	private static final long serialVersionUID = 1L;
	
	public enum ACTIVATION {
		SIGMOID,
		RELU,
		TANH,
		SOFTPLUS,
		BSTEP,
		SOFTMAX,
		LINEAR;
	}
		
	private String orientation;
		
	private final List<Dense> denseLayers;
	private final List<Conv2D> convLayers;
	
	Optimizer opt;
	
	public abstract class Options {
				
		private boolean regularized;
		private float regFactor;
		
		private boolean onlyUpdateSelected;
		private int[] toUpdate;
		
		public String orientation;
		
		public float lr;
		public float valSplit;
		
		public int epochs;
		public int batchSize;
		
		public boolean shuffle;
		public boolean saveIterLog;
		public boolean saveEpochLog;
		
		private Sequential parent;
		
		public Options(Sequential parent) {
			
			regularized = false;
			onlyUpdateSelected = false;
			
			orientation = "horizontal";
			
			lr = 0.001f;
			valSplit = 0.0f;
			
			epochs = 1;
			batchSize = 1;
			
			shuffle = false;
			saveIterLog = false;
			saveEpochLog = true;
			
			this.parent = parent;
		}
		
		public abstract void tweak();
		
		public void onTrainingEnd() {return;}
		public void onEpochEnd(int epoch, float trainLoss, float valLoss, String epochLog) {return;}
		public void onIterEnd(int iter, float trainLoss, float valLoss, String iterLog) {return;}
		public void onTrainingStart() {return;}
		public final void regularize(float factor) {regularized = true; regFactor = factor;}
		public final void onlyUpdateSelected(int[] updates) {onlyUpdateSelected = true; toUpdate = updates;}
		public final boolean isRegularized() {return regularized;}
		public final float getRegFactor() {return regFactor;}
		public final boolean onlySelected() {return onlyUpdateSelected;}		
		public final int[] updates() {return toUpdate;}
		public final Sequential getParent() {return parent;}
	}
	
	public Sequential() {
		
		denseLayers = new ArrayList<Dense>();
		convLayers = new ArrayList<Conv2D>();
		
		opt = null;
		orientation = "horizontal";
	}
	
	public Sequential(Sequential copied) {
		
		this();
		
		for (Conv2D conv : copied.getConv())
			convLayers.add(conv.copy());
		
		for (Dense dense : copied.getDense())
			denseLayers.add(dense.copy());
		
		orientation = copied.getOrientation();
	}
	
	public Sequential add(Dense addit) {
		
		if (hasDense())
			addit.init(getLastDense().getNeurons());
		else if (hasConv())
			addit.init(getLastConv().getPooledOutSize());
		else if (!addit.isReady())
			throw new RuntimeException("Must specify input shape for the first layer");
		else if (addit.getNeurons() == 0)
			throw new RuntimeException("Cannot add empty dense layer");
		
		getDense().add(addit);
		return this;
	}
	
	public Sequential add(Conv2D addit) {
		
		if (hasDense())
			throw new RuntimeException("Cannot add a conv layer after a dense layer: Output is flattened!");
		
		if (hasConv())
			addit.init(getLastConv().getPooledOutShape());
		else if (!addit.isReady())
			throw new RuntimeException("Must specify input shape for the first layer");
		else if (addit.getFilterCount() == 0)
			throw new RuntimeException("Cannot add empty conv layer");
		
		getConv().add(addit);
		return this;
	}
	
	public Sequential remove(int index) {
		
		if (index >= getCount())
			throw new RuntimeException("Index exceeds number of layers in remove()");
		
		if (index >= getConvCount())
			getDense().remove(index - getConvCount());
		else
			getConv().remove(index);
		return this;
	}
	
	public Sequential removeDense(int index) {
		
		if (index >= getDenseCount())
			throw new RuntimeException("Index exceeds number of dense layers in removeDense()");
		
		getDense().remove(index);
		return this;
	}
	
	public Sequential removeConv(int index) {
		
		if (index >= getConvCount())
			throw new RuntimeException("Index exceeds number of conv layers in removeConv()");
		
		getConv().remove(index);
		return this;
	}
	
	public void clear() {
		
		getConv().clear();
		getDense().clear();
	}
	
	public Sequential pool(Conv2D.POOLING pool, int stride) {
		
		if (getDenseCount() > 0)
			throw new RuntimeException("Cannot add pool to a Dense layer");
		
		getLastConv().setPool(pool, stride);
		getLastConv().init(getLastConv().getInputShape());
		return this;
	}
	
	public Sequential padding(Conv2D.PADDING pad) {
		
		if (getDenseCount() > 0)
			throw new RuntimeException("Cannot add padding to a Dense layer");
		
		getLastConv().setPad(pad);
		getLastConv().init(getLastConv().getInputShape());
		return this;
	}
	
	public Sequential orientation(String or) {
		
		if (or.equals("horizontal") || or.equals("vertical"))
			orientation = or;
		else
			throw new RuntimeException("Unknow orientation: " + or);
		return this;
	}
	
	public Vector feedForward(Vector input) {
		
		if (isEmpty())
			return input;
		
		validateInput(input);
		
		Vector output;
		if (hasConv()) {
			
			Matrix[] convOut = getFirstConv().toInputVolume(input, getOrientation());
			for (Conv2D conv : getConv())
				convOut = conv.computeOutput(convOut);
			
			if (isHorizontal())
				output = Matrix.appendHor(convOut).flatten();
			else
				output = Matrix.appendVer(convOut).flatten();
			
			getLastConv().setFlat(output.copy());
		} else
			output = input;
		
		for (Dense dense : getDense())
			output = dense.computeOutput(output);
		
		return output;
	}
	
	private void validateInput(Vector input) {
		
		if (isEmpty())
			return;
		
		if (hasConv()) {
			if (getFirstConv().getInputSize() != input.getLength())
				throw new RuntimeException("Input dimensions mismatch with the network input shape");
				
		} else if (getFirstDense().getInputShape() != input.getLength())
			throw new RuntimeException("Input dimensions mismatch with the network input shape");
		return;
	}
	
	public Sequential optimizer(Optimizer.LOSS loss) {
		
		opt = new Optimizer(this, loss);
		return this;
	}
	
	public Sequential fit(Vector[] trainSet, Vector[] labelSet) {
		
		Options options = new Options(this) {
			@Override
			public void tweak() {
				return;
			}
		};
		return fit(trainSet, labelSet, options);
	}
	
	public Sequential fit(Vector[] trainSet, Vector[] labelSet, float learningRate, int bSize, int nEpochs, float validationSplit, boolean isShuffled) {
		
		Options options = new Options(this) {
			@Override
			public void tweak() {
				lr = learningRate;
				batchSize = bSize;
				epochs = nEpochs;
				valSplit = validationSplit;
				shuffle = isShuffled;
			}
		};
		
		return fit(trainSet, labelSet, options);
	}
	
	public Sequential fit(Vector[] trainSet, Vector[] labelSet, Options options) {
		
		options.tweak();
		options.onTrainingStart();
		
		if (getOpt() == null)
			throw new RuntimeException("Must set the optimizer befor calling fit()");
		
		if (trainSet.length != labelSet.length)
			throw new RuntimeException("Train set and label set length must be equal");
		
		if (options.batchSize < 1)
			throw new RuntimeException("Batch size must be greater than 0");
		
		if (options.valSplit >= 1.0f)
			throw new RuntimeException("The validation split must be smaller than 1");
		
		List<Integer> indexes = new ArrayList<Integer>(trainSet.length);
		List<Float> splits = new ArrayList<Float>(trainSet.length);
		
		for (int i = 0; i < trainSet.length; i++) {
			indexes.add(i);
			splits.add(1.0f - i * 1.0f / trainSet.length);
		}
		
		Collections.shuffle(splits);
		
		for (int i = 0; i < options.epochs; i++) {
			float trainLoss = 0.0f;
			float valLoss = 0.0f;
						
			int trainCount = 0;
			int valCount = 0;
			
			if (options.shuffle)
				Collections.shuffle(indexes);
						
			for (int j = 0; j < trainSet.length; j++) {
				
				int index = indexes.get(j);
				float split = splits.get(index);
				
				Vector inputSample = trainSet[index];
				
				Vector label = labelSet[index];
				Vector guess = feedForward(inputSample);
				
				float loss = getOpt().computeLoss(guess, label);
				if (split >= options.valSplit) {
					
					trainLoss += loss;
					
					Vector lossDeriv = getOpt().computeLossDeriv(guess, label);
					if (options.onlySelected())
						lossDeriv.setAllBut(options.updates()[index], 0.0f);
					
					getOpt().backpropagate(inputSample, lossDeriv);
					
					trainCount++;
					if (trainCount % options.batchSize == 0) {
						getOpt().fix(options);
						getOpt().zeroGrad();
					}
					
				} else {
					valLoss += loss;
					valCount++;
				}
								
				String iterLog = "";
				if (options.saveIterLog)
					iterLog = ("Epoch: " + String.valueOf(i + 1) + "/" + options.epochs + 
							" Training progress: " + Math.round(100 * 1000 * (float) j / trainSet.length) / 1000.0 + "%" +
							" Training loss: " + String.valueOf((float) trainLoss / (trainCount > 0 ? trainCount : 1)) + " Validation loss: " + String.valueOf((float) valLoss / (valCount > 0 ? valCount : 1)));
				
				options.onIterEnd(j + 1, trainCount > 0 ? trainLoss / trainCount : 0.0f, valCount > 0 ? valLoss / valCount : 0.0f, iterLog);
			}
			
			String epochLog = "";
			
			trainLoss /= trainCount > 0 ? trainCount : 1;
			valLoss /= valCount > 0 ? valCount : 1;
			if (options.saveEpochLog)
					epochLog ="Epoch " + String.valueOf(i + 1) + "/" + String.valueOf(options.epochs) + " Train loss: " + String.valueOf(trainLoss)
					+ " Validation loss: " + String.valueOf(valLoss);
			
			options.onEpochEnd(i + 1, trainLoss, valLoss, epochLog);
		}
		
		options.onTrainingEnd();
		return this;
	}
	
	public Sequential summary() {
		
		int count = 1;
		int totalTrainable = 0;
		
		System.out.println("======Model summary======");
		System.out.println("----");
		System.out.println("Convolutional layers: " + getConvCount());
		System.out.println("Dense layers: " + getDenseCount());
		System.out.println("----");
		
		for (Conv2D conv : getConv()) {
			System.out.println(count++ + ": Conv2D. Activation: " + conv.getLabel().toString());
			System.out.println("---InputShape(" + conv.getInputShape()[0] + ", " + conv.getInputShape()[1] + ", " + conv.getInputShape()[2] + ") "
					 + "OutputShape(" + conv.getOutputShape()[0] + ", " + conv.getOutputShape()[1] + ", " + conv.getOutputShape()[2] + ") "
					 + "Trainable Params: #" + conv.getFilterSize() * conv.getFilterCount() + "---");
			
			if (conv.getPool() != Conv2D.POOLING.NONE) {
				System.out.println(count++ + ": Pooling " + conv.getPool().toString());
				System.out.println("---InputShape(" + conv.getOutputShape()[0] + ", " + conv.getOutputShape()[1] + ", " + conv.getOutputShape()[2] + ") "
						 + "OutputShape(" + conv.getPooledOutShape()[0] + ", " + conv.getPooledOutShape()[1] + ", " + conv.getPooledOutShape()[2] + ") "
						 + "Trainable Params: #0---");
			}
			
			totalTrainable += conv.getFilterSize() * conv.getFilterCount();
		}
		
		for (Dense dense : getDense()) {
			System.out.println(count++ + ": Dense. Activation: " + dense.getLabel().toString());
			System.out.println("---InputShape(" + dense.getInputShape() + ", 1, 1) "
					 + "OutputShape(" + dense.getNeurons() + ", 1, 1) "
					 + "Trainable Params: #" + dense.getInputShape() * dense.getNeurons() + "---");
			
			totalTrainable += dense.getInputShape() * dense.getNeurons();
		}
		
		System.out.println("Total Trainable Params: " + totalTrainable);
		System.out.println("======End======");
		return this;
	}
	
	public Sequential randomize(float deviation) {
		
		for (Dense dense : getDense())
			dense.randomize(deviation);
		
		for (Conv2D conv : getConv())
			conv.randomize(deviation);	
		return this;
	}
	
	public Sequential saveModel(String sel) {
		
		try {
			FileOutputStream fileOut = new FileOutputStream(new File(sel));
	        ObjectOutputStream objectOut = new ObjectOutputStream(fileOut);
	        
	        objectOut.writeObject(this);
	        
	        objectOut.close();
	        fileOut.close();
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return this;
	}
	
	public static Sequential loadModel(String sel) {
		
		try {
			FileInputStream fileIn = new FileInputStream(new File(sel));
			ObjectInputStream objectIn = new ObjectInputStream(fileIn);
			
			Sequential result = (Sequential) objectIn.readObject();
			
			objectIn.close();
			fileIn.close();
			
			return result;
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			throw new RuntimeException(e);
		} catch (ClassNotFoundException e) {
			throw new RuntimeException(e);
		}
	}
	
	public PImage[][] getAllFeatureMaps(){
		
		if (!hasDense())
			throw new RuntimeException("No convolutional layers found");
		
		PImage[][] images = new PImage[getConvCount()][];
		for (int i = 0; i < getConvCount(); i++)
			images[i] = getFeatureMaps(i);
		
		return images;
	}
	
	public PImage[] getFeatureMaps(int index) {
		
		if (!hasDense())
			throw new RuntimeException("No convolutional layers found");
		
		if (index >= getConvCount())
			throw new RuntimeException("Index exceeds number of convolutional layers");
		
		Conv2D conv = getConv().get(index);
		PImage[] images = new PImage[conv.getOutputShape()[2]];
		for (int i = 0; i < images.length; i++) {
			images[i] = createImage(conv.getOutputShape()[1], conv.getOutputShape()[2], RGB);
			images[i].loadPixels();
			
			for (int j = 0; j < images[i].height; j++)
				for (int k = 0; k < images[i].width; k++)
					images[i].set(k, j, color(conv.getOutVal()[i].get(j, k) * 255));
		}
		
		return images;
	}
	
	public List<Dense> getDense() {return denseLayers;}
	public List<Conv2D> getConv() {return convLayers;}
	public Dense getLastDense() {return denseLayers.get(denseLayers.size() - 1);}
	public Conv2D getLastConv() {return convLayers.get(convLayers.size() - 1);}
	public Dense getFirstDense() {return denseLayers.get(0);}
	public Conv2D getFirstConv() {return convLayers.get(0);}
	public int getDenseCount() {return denseLayers.size();}
	public int getConvCount() {return convLayers.size();}
	public boolean hasDense() {return !getDense().isEmpty();}
	public boolean hasConv() {return !getConv().isEmpty();}
	public boolean isEmpty() {return !hasDense() && !hasConv();}
	public String getOrientation() {return orientation;}
	public boolean isHorizontal() {return orientation.equals("horizontal");}
	public boolean isVertical() {return orientation.equals("vertical");}
	public int getCount() {return getDenseCount() + getConvCount();}
	public Optimizer getOpt() {return opt;}
	
	public static List<ACTIVATION> getActLabels() {
		List<ACTIVATION> labels = new ArrayList<ACTIVATION>();
		labels.add(ACTIVATION.SIGMOID);
		labels.add(ACTIVATION.RELU);
		labels.add(ACTIVATION.TANH);
		labels.add(ACTIVATION.SOFTPLUS);
		labels.add(ACTIVATION.BSTEP);
		labels.add(ACTIVATION.SOFTMAX);
		labels.add(ACTIVATION.LINEAR);
		return labels;
	}
	
}
