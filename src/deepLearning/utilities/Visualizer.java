package deepLearning.utilities;

import processing.core.*;
import java.util.List;
import java.util.ArrayList;

public class Visualizer {
	
	private List<List<PVector>> convPos;
	private List<List<PVector>> densePos;
		
	private Sequential model;
	private PApplet parent;
	
	public int inputCol, hiddenCol, outputCol, textCol, textLabelCol;
	public int size, spaceForText, textSize;
	
	private String[] outputLabels;
	private String[] inputLabels;
	
	public Visualizer(PApplet parent, Sequential model) {
		
		this.parent = parent;
		this.model = model;
		
		convPos = new ArrayList<List<PVector>>(model.getConvCount() + 1);
		densePos = new ArrayList<List<PVector>>(model.getDenseCount() + 1);
		
		inputCol = parent.color(0, 255, 0);
		hiddenCol = parent.color(255, 255, 0);
		outputCol = parent.color(255, 0, 0);
		textCol = parent.color(255);
		textLabelCol = parent.color(0);
		
		size = 20;
		spaceForText = 40;
		textSize = 15;
		
		outputLabels = null;
		inputLabels = null;
	}
	
	public void tweak() {
		return;
	}
	
	public Visualizer updateAsModel() {
		
		if (model.isEmpty())
			throw new RuntimeException("Cannot update: the model is empty. Add at least an output layer");
		
		convPos.clear();
		densePos.clear();
		
		float left = 3 * size;
		float right = parent.width - 3 * size;
		
		float up = size;
		float down = parent.height - size - spaceForText;
		
		float x = left;
		float dx = (right - left) / model.getCount();
		
		int inputNodes;
		List<PVector> inputList = new ArrayList<PVector>();
		
		if (model.hasConv()) {
			inputNodes = model.getFirstConv().getInputSize();
			convPos.add(inputList);
		}
		else {
			inputNodes = model.getFirstDense().getInputShape();
			densePos.add(inputList);
		}
		
		float dyInput = (down - up) / (inputNodes + 1);
		float yInput = up;
		
		for (int j = 0; j < inputNodes; j++)
			inputList.add(new PVector(x, yInput += dyInput));
			
				
		for (int i = 0; i < model.getConvCount(); i++) {
			Conv2D conv = model.getConv().get(i);
			List<PVector> list = new ArrayList<PVector>();
			
			x += dx;
			
			float dy = (down - up) / (conv.getFilterCount() + 1);
			float y = up;
			for (int j = 0; j < conv.getFilterCount(); j++)
				list.add(new PVector(x, y += dy));
			
			convPos.add(list);
		}
		
		for (int i = 0; i < model.getDenseCount(); i++) {
			Dense dense = model.getDense().get(i);
			List<PVector> list = new ArrayList<PVector>();
			
			x += dx;
			
			float dy = (down - up) / (dense.getNeurons() + 1);
			float y = up;
			for (int j = 0; j < dense.getNeurons(); j++)
				list.add(new PVector(x, y += dy));
			
			densePos.add(list);
		}
		return this;
	}
	
	public Visualizer draw() {
		
		if (isEmpty())
			throw new RuntimeException("Cannot draw: must call updateAsModel() first");
		
		if (model.isEmpty())
			throw new RuntimeException("Cannot draw: the model is empty. Add at least an output layer");
		
		if (outputLabels != null && !model.hasDense())
			throw new RuntimeException("Cannot use output labels with conv layers as outputs");
		
		if (inputLabels != null && model.hasConv())
			throw new RuntimeException("Cannot use input labels with conv layers");
		
		if (outputLabels != null && outputLabels.length != model.getLastDense().getNeurons())
			throw new RuntimeException("Output labels array length mismatch with number of nodes in the output layer");
		
		if (inputLabels != null && inputLabels.length != model.getFirstDense().getInputShape())
			throw new RuntimeException("Input labels array length mismatch with model's input shape");
		
		parent.push();
		
		parent.textAlign(PApplet.CENTER);
		parent.textSize(textSize);
		parent.rectMode(PApplet.CENTER);
		
		parent.stroke(255);
		
		for (int i = 0; i < convPos.size() - 1; i++) {
			
			List<PVector> current = convPos.get(i);
			List<PVector> next = convPos.get(i + 1);
			for (PVector currentPos : current)
				for (PVector nextPos : next)
					parent.line(currentPos.x, currentPos.y, nextPos.x, nextPos.y);
		}
		
		for (int i = 0; i < densePos.size() - 1; i++) {
			
			List<PVector> current = densePos.get(i);
			List<PVector> next = densePos.get(i + 1);
			for (PVector currentPos : current)
				for (PVector nextPos : next)
					parent.line(currentPos.x, currentPos.y, nextPos.x, nextPos.y);
		}
		
		parent.stroke(0);
		
		int index = - 1;
		for (List<PVector> list : convPos) {
			
			if (index == - 1)
				parent.fill(inputCol);
			else if (index == model.getConvCount() - 1)
				parent.fill(outputCol);
			else
				parent.fill(hiddenCol);
			
			int i = 0;
			int j = 0;
			for (PVector pos : list) {
				parent.rect(pos.x, pos.y, 2 * size, 2 * size);
				
				parent.push();
				parent.fill(textLabelCol);
				if (outputLabels != null && index == model.getConvCount() - 1)
					parent.text(outputLabels[i++], pos.x, pos.y);
				else if (inputLabels != null && index == - 1)
					parent.text(inputLabels[j++], pos.x, pos.y);
				
				parent.pop();
			}
			
			if (index > - 1) {
				parent.fill(textCol);
				parent.text("Conv2D " + "(" +  model.getConv().get(index).getFilterCount() + ")" + ": " + model.getConv().get(index).getLabel().toString(), list.get(0).x, parent.height - spaceForText / 2);
			}
			index++;
		}
		
		index = - 1;
		for (List<PVector> list : densePos) {
			
			if (index == - 1)
				parent.fill(inputCol);
			else if (index == model.getDenseCount() - 1)
				parent.fill(outputCol);
			else
				parent.fill(hiddenCol);
			
			int i = 0;
			int j = 0;
			for (PVector pos : list) {
				parent.circle(pos.x, pos.y, 2 * size);
				
				parent.push();
				parent.fill(textLabelCol);
				if (outputLabels != null && index == model.getDenseCount() - 1)
					parent.text(outputLabels[i++], pos.x, pos.y);
				else if (inputLabels != null && index == - 1)
					parent.text(inputLabels[j++], pos.x, pos.y);
				
				parent.pop();
			}
			
			if (index > - 1) {
				parent.fill(textCol);
				parent.text("Dense " + "(" + model.getDense().get(index).getNeurons() + ")" + ": " + model.getDense().get(index).getLabel().toString(), list.get(0).x, parent.height - spaceForText / 2);
			}
			index++;
		}
		
		parent.pop();
		return this;
	}
	
	public Visualizer setInputLabels(String[] inputLabels) {
		
		this.inputLabels = inputLabels;
		return this;
	}
	
	public Visualizer setInputLabels(List<String> labels) {
		
		inputLabels = new String[labels.size()];
		 for (int i = 0; i < labels.size(); i++)
			 inputLabels[i] = labels.get(i);
		 return this;
	}
	
	public Visualizer setOutputLabels(String[] outputLabels) {
		
		this.outputLabels = outputLabels;
		return this;
	}
	
	public Visualizer setOutputLabels(List<String> labels) {
		
		 outputLabels = new String[labels.size()];
		 for (int i = 0; i < labels.size(); i++)
			 outputLabels[i] = labels.get(i);
		 return this;
	}
	
	public boolean isEmpty() {
		
		return convPos.isEmpty() && densePos.isEmpty();
	}
	
	public List<List<PVector>> getPos(){
		
		List<List<PVector>> all = new ArrayList<List<PVector>>();
		for (List<PVector> convList : convPos)
			all.add(convList);
		
		for (List<PVector> denseList : densePos)
			all.add(denseList);
		
		return all;
	}
	
	public List<PVector> getPos(int index) {
		
		if (index >= convPos.size() + densePos.size())
			throw new RuntimeException("Index exceeds number of layers");
		
		if (index >= convPos.size())
			return densePos.get(index - convPos.size());
		
		return convPos.get(index);
	}
	
	public Sequential getModel() {
		
		return model;
	}
	
	public Visualizer setModel(Sequential model) {
		
		this.model = model;
		
		inputLabels = null;
		outputLabels = null;
		return this;
	}
}
