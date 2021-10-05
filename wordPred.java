/*
 * Project: Next Word Predictor using N-grams: Allow user to enter a word
 * 			and predict a set of possible next words using Add one smoothing and Good Turing methods
 * 
*/

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.math.BigDecimal;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.NavigableMap;
import java.util.Scanner;
import java.util.Set;
import java.util.SortedSet;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.TreeSet;

class BigramWords{
	private String currWord;
	private String nextWord;

	public BigramWords(String currWord, String nextWord) {
		super();
		this.currWord = currWord;
		this.nextWord = nextWord;
	}

	public String getCurrWord() {
		return currWord;
	}

	public void setCurrWord(String currWord) {
		this.currWord = currWord;
	}

	public String getNextWord() {
		return nextWord;
	}

	public void setNextWord(String nextWord) {
		this.nextWord = nextWord;
	}

	public static BigramWords getBigramNode(String currentWord, String folWord) {
		return new BigramWords(currentWord, folWord);
	}

	public int hashCode() {
		return currWord.hashCode() + nextWord.hashCode();
	}

	@Override
	public boolean equals(Object otherBigramWord) 
	{
		if (otherBigramWord == null || !(otherBigramWord instanceof BigramWords)) {
			return false;
		}

		if (this.getCurrWord().equals(((BigramWords) otherBigramWord).getCurrWord()) 
				&& this.getNextWord().equals(((BigramWords) otherBigramWord).getNextWord())) {
			return true;
		}

		return false;
	}

	@Override
	public String toString() {
		return getNextWord() + "|" + getCurrWord();
	}
}

public class wordPred {

	private Map<String,Double> unigrams;
	private Map<BigramWords,Double> bigrams;
	private TreeSet<Map.Entry<BigramWords, Double>> predictingBigrams;
	private Map<String,Double> sentencesInCorpus;
	private Map<BigramWords,Double> probTable;
	private double vocabSize;
	long cleanedSentences;

	private Map<Double,Double> goodTuringFreq;

	private File file;
	
	//Replace special character with "" (no character)
	private Set<String> ignoredTokens = new HashSet<String>() {{ 
		add("`"); add("''"); add("'"); add("."); add(";"); add("``"); 
	}};

	//Constructor
	public wordPred()
	{
		sentencesInCorpus = new HashMap<String,Double>();
		unigrams = new HashMap<String,Double>();
		bigrams = new HashMap<BigramWords,Double>();
		probTable = new HashMap<BigramWords,Double>();
		goodTuringFreq = new HashMap<Double,Double>();
		predictingBigrams = new TreeSet<>(new Comparator<Map.Entry<BigramWords, Double>>() {

			@Override
			public int compare(Entry<BigramWords, Double> o1, Entry<BigramWords, Double> o2) {
				// TODO Auto-generated method stub
				return o1.getValue().compareTo(o2.getValue());
			}
		});
		vocabSize = 0.0;
	}

	//Reading Corpus file and removing all special characters
	public void readCorpusFile(String fName)
	{
		try{
			System.out.println("Cleaning of Corpus.txt File.");

			File corpusFile = new File(fName);
			Scanner sc_corpusFile = new Scanner(corpusFile);
			StringBuffer sb_corpusBuffer = new StringBuffer();

			long totalNumOfSentences = 0;
			String line = null;

			//Read file and scan and replace all non word sequences
			while(sc_corpusFile.hasNextLine())
			{
				line = sc_corpusFile.nextLine();
				line = line.replaceAll("[^\\w\\s]","");
				line = line.toLowerCase();
				sb_corpusBuffer.append(" ").append(line);
			}
			
			//Save the cleaned corpus as cleanedCorpus.txt
			File file = new File("cleanedCorpus.txt");
			BufferedWriter bwr=  new BufferedWriter(new FileWriter(file));
			bwr.write(sb_corpusBuffer.toString());
			bwr.close();
			String regex = "^\\s+$";
			StringTokenizer stringToken = new StringTokenizer(sb_corpusBuffer.toString(), ",");

			while(stringToken.hasMoreTokens())
			{
				String corpusSentence = (String) stringToken.nextElement();

				totalNumOfSentences++;

				if (sentencesInCorpus.containsKey(corpusSentence))
					sentencesInCorpus.put(corpusSentence,sentencesInCorpus.get(corpusSentence) + 1);
				else
					sentencesInCorpus.put(corpusSentence, 1.0);
			}

			cleanedSentences = totalNumOfSentences;
			System.out.println("Cleaning of Corpus completed.");

			sc_corpusFile.close();
		}
		catch(Exception e){
			e.getMessage().toString();
		}
	}

	private List<String> removeIgnoredTokensFromInput(String sentence)
	{
		StringTokenizer sentTokenizer = new StringTokenizer(sentence, " ");
		List<String> validSentenceTokens = new ArrayList<String>();

		while(sentTokenizer.hasMoreElements())
		{
			String sentToken = (String) sentTokenizer.nextElement();

			if(!ignoredTokens.contains(sentToken)){
				validSentenceTokens.add(sentToken);
			}
		}

		return validSentenceTokens;
	}

	//Compute the Local Bigram Model
	private Map<BigramWords, Double> computeLocBigramModel(String sentence, Double occ)
	{
		Map<BigramWords, Double> tempTableMap = new LinkedHashMap<BigramWords, Double>();
		List<String> validSentenceTokens = removeIgnoredTokensFromInput(sentence);

		for(int index = 0; index < validSentenceTokens.size() - 1; index++)
		{
			BigramWords node = new BigramWords(validSentenceTokens.get(index).toLowerCase(), validSentenceTokens.get(index + 1).toLowerCase());

			if(tempTableMap.containsKey(node)){
				tempTableMap.put(node, tempTableMap.get(node) + 1 * occ);
			}

			else{
				tempTableMap.put(node, 1 * occ);
			}
		}

		for(String validToken : validSentenceTokens)
		{
			unigrams.put(validToken.toLowerCase(), (unigrams.containsKey(validToken) ? unigrams.get(validToken) + 1 : 1));
		}

		return tempTableMap;
	} 

	//Generate Bigram Model for Corpus 
	public void computeBigrams()
	{
		Set<Map.Entry<String, Double>> entryCorpusSet = sentencesInCorpus.entrySet();

		for(Map.Entry<String, Double> entry : entryCorpusSet )
		{
			String sentence = entry.getKey();
			Double occurence = entry.getValue();

			Map<BigramWords,Double> bigramMap= computeLocBigramModel(sentence,occurence);
			Set<Map.Entry<BigramWords, Double>> locBigramEntries = bigramMap.entrySet();

			for (Map.Entry<BigramWords, Double> localBigramEntry : locBigramEntries) 
			{
				BigramWords keyBigramNode = localBigramEntry.getKey();
				Double localOcc = localBigramEntry.getValue();

				if (bigrams.containsKey(keyBigramNode)) 
					bigrams.put(keyBigramNode, bigrams.get(keyBigramNode) + localOcc);
				else 
					bigrams.put(keyBigramNode, localOcc);
			}	
		}
	}

	//Display the Bigram model for Corpus
	public void displayCorpusBigramMod()
	{
		Set<Entry<BigramWords, Double>> bigramEntries = bigrams.entrySet();
		for(Entry<BigramWords, Double> globalBigramEntry : bigramEntries)
		{
			BigramWords bigramNode = globalBigramEntry.getKey();
			Double occs = globalBigramEntry.getValue();

			System.out.println("Bigram Node Key: " + bigramNode + "Occurrences: " + occs);
		}

		System.out.println("Size of Bigram Entries" + bigramEntries.size());
	}

	private String getTabbedCol(String string)
	{
		return string.length() > 7 ? "\t\t" : "\t";
	}

	//Display Bigram Count Model for Corpus
	public void displayInputBigramCountMod(String ipLine)
	{
		List<String> validSentenceTokens = removeIgnoredTokensFromInput(ipLine);
		StringBuffer sb_dispScreen = new StringBuffer();

		int maxTokLen = 0;

		for(String validSentenceToken : validSentenceTokens)
		{
			if(validSentenceToken.length() > maxTokLen){
				maxTokLen = validSentenceToken.length();
			}
		}

		String tab = (maxTokLen > 7 ? "\t\t" : "\t");
		sb_dispScreen.append(tab);

		for(int index = 0; index < validSentenceTokens.size(); index++)
		{
			sb_dispScreen.append(validSentenceTokens.get(index)).append("\t");
		}

		for(int rowIndex = 0; rowIndex < validSentenceTokens.size(); rowIndex++)
		{
			String currWord = validSentenceTokens.get(rowIndex);
			sb_dispScreen.append("\n").append(currWord).append(currWord.length() <= 7 ? tab : "\t");

			for (int colIndex = 0; colIndex < validSentenceTokens.size(); colIndex ++)
			{
				String folWord = validSentenceTokens.get(colIndex);
				Double cntVal = bigrams.get(BigramWords.getBigramNode(currWord, folWord));

				sb_dispScreen.append((cntVal == null ? 0 : cntVal)).append(getTabbedCol(folWord));
			}	
		}
		System.out.println(sb_dispScreen.toString());
	}

	//Compute All the Bigram Models Possible
	public List<Map<BigramWords, Double>> computeAllBigramModels(String sentence)
	{
		List<Map<BigramWords, Double>> allBigramModels = new LinkedList<Map<BigramWords, Double>>();
		Map<BigramWords, Double> possibleModel = null;

		List<String> validSentTokens = removeIgnoredTokensFromInput(sentence);

		for(int rowIndex = 0; rowIndex < validSentTokens.size(); rowIndex++)
		{
			String currWord = validSentTokens.get(rowIndex);
			possibleModel = new LinkedHashMap<BigramWords, Double>();

			for (int columnIndex = 0; columnIndex < validSentTokens.size(); columnIndex ++)
			{
				String folWord = validSentTokens.get(columnIndex);
				BigramWords possibleBigramNode = BigramWords.getBigramNode(currWord, folWord);
				Double count = null;

				if (bigrams.containsKey(possibleBigramNode)) 
				{
					count = bigrams.get(possibleBigramNode);
				} else {
					count = 0.0;
				}
				possibleModel.put(possibleBigramNode, Double.valueOf(count));
			}
			allBigramModels.add(possibleModel);
		}
		return allBigramModels;
	}

	//Display the Probability Model for the Smoothing
	public void dispProbModel(List<Map<BigramWords, Double>> computedBigramModel)
	{
		if (computedBigramModel == null || computedBigramModel.size() == 0)
		{
			return;
		}

		StringBuffer sb = new StringBuffer();
		List<String> validSentTokens = new LinkedList<String>();
		int maxTokenLen = 0;

		for (Map.Entry<BigramWords, Double> bigramModel : computedBigramModel.get(0).entrySet()) 
		{
			String sentToken = bigramModel.getKey().getNextWord();
			validSentTokens.add(sentToken);

			if (sentToken.length() > maxTokenLen)
				maxTokenLen = sentToken.length();
		}

		String tab = (maxTokenLen > 8 ? "\t\t" : "\t");
		sb.append(tab);

		for (int index = 0; index < validSentTokens.size(); index++) 
		{
			sb.append(validSentTokens.get(index)).append("\t");
		}

		for (int rowIndex = 0; rowIndex < validSentTokens.size(); rowIndex ++) 
		{
			String currWord = validSentTokens.get(rowIndex);
			sb.append("\n").append(currWord).append(currWord.length() <= 8 ? tab : "\t");

			for (int colIndex = 0; colIndex < validSentTokens.size(); colIndex ++) 
			{
				String folWord = validSentTokens.get(colIndex);

				Map<BigramWords, Double> bigramModel = computedBigramModel.get(rowIndex);

				Double probablityValue = bigramModel.get(BigramWords.getBigramNode(currWord, folWord));
				sb.append(probablityValue).append(getTabbedCol(folWord));
			}	
		}	
		System.out.println(sb.toString());
	}

	private void computeBigramModelWithoutSmoothing(String ipLine)
	{

		predictingBigrams.clear();
		String previousWords[] = ipLine.split(" ");
		String previousWord = previousWords[previousWords.length -1];
		Set<Map.Entry<BigramWords, Double>> bigramModelEntries = bigrams.entrySet();
		for (Map.Entry<BigramWords, Double> bigramModelEntry : bigramModelEntries) 
		{
			BigramWords locBigramNode  = bigramModelEntry.getKey();

			Double probWithoutSmoothing = 0.0;

			if (unigrams.containsKey(locBigramNode.getCurrWord()) && locBigramNode.getCurrWord().equalsIgnoreCase(previousWord)) 
			{
				probWithoutSmoothing = bigramModelEntry.getValue() / unigrams.get(locBigramNode.getCurrWord());
				probWithoutSmoothing = new BigDecimal(probWithoutSmoothing).setScale(5, BigDecimal.ROUND_HALF_UP).doubleValue();
				predictingBigrams.add(new AbstractMap.SimpleEntry<BigramWords,Double>(locBigramNode,probWithoutSmoothing) {
				});
			}
		}

		predictNextWord();
	}

	//Predict the next word based on the probability
	private  void predictNextWord(){
		int length = 0;
		length = predictingBigrams.size() >=5 ? 5 :predictingBigrams.size();
		System.out.println("length is " + length);
		if(length > 0){
			for(int i = 0; i< length; i++){
				Entry<BigramWords,Double> lastEntry  = predictingBigrams.last();

				String nextWord = lastEntry.getKey().getNextWord();
				System.out.println(nextWord);
				predictingBigrams.remove(predictingBigrams.last());
			}
		} 
		else System.out.print("no word found");
	}

	private void computeBigramModelWithAddOne(String ipLine)
	{
		predictingBigrams.clear();
		String previousWords[] = ipLine.split(" ");
		String previousWord = previousWords[previousWords.length -1];
		Set<Map.Entry<BigramWords, Double>> bigramModelEntries = bigrams.entrySet();
		for (Map.Entry<BigramWords, Double> bigramModelEntry : bigramModelEntries) 
		{
			BigramWords bigram  = bigramModelEntry.getKey();
			Double bigramOcc = bigramModelEntry.getValue();
			Double addOneSmoothing = 0d;

			if (unigrams.containsKey(bigram.getCurrWord()) && bigram.getCurrWord().equalsIgnoreCase(previousWord))
			{
				Double prevWordTokenOcc = unigrams.get(bigram.getCurrWord());
				addOneSmoothing = (bigramOcc + 1.0) / (prevWordTokenOcc + unigrams.size());
				addOneSmoothing = new BigDecimal(addOneSmoothing).setScale(5, BigDecimal.ROUND_HALF_UP).doubleValue();
				predictingBigrams.add(new AbstractMap.SimpleEntry(bigram, addOneSmoothing));
			}
		}
		predictNextWord();
	}

	public void genGoodTuringFreq(){
		for(Double occVal : bigrams.values()){
			if(goodTuringFreq.containsKey(occVal)){
				goodTuringFreq.put(occVal, goodTuringFreq.get(occVal) + 1);
			}
			else{
				goodTuringFreq.put(occVal, 1.0);
			}
		}
	}

	public void computeBigramModelWithGoodTuringSmoothing(String ipLine)
	{
		genGoodTuringFreq();
		predictingBigrams.clear();
		String previousWords[] = ipLine.split(" ");
		String previousWord = previousWords[previousWords.length -1];
		Double totNumOfWordsInVoc = 0.0;
		for (Double cnt : unigrams.values())
		{
			totNumOfWordsInVoc += cnt;
		}

		Set<Map.Entry<BigramWords, Double>> bigramModelEntries = bigrams.entrySet();

		for (Map.Entry<BigramWords, Double> bigramModelEntry : bigramModelEntries) 
		{

			BigramWords bigram = bigramModelEntry.getKey(); 
			Double goodTuringSmoothing = 0.0;
			if(bigram.getCurrWord().equalsIgnoreCase(previousWord)){
				Double bigramOcc = bigramModelEntry.getValue();
				if (bigramOcc == 0) 
				{
					goodTuringSmoothing = (1.0 * (goodTuringFreq.get(1.0) / totNumOfWordsInVoc));  
				} 
				else 
				{
					if (goodTuringFreq.containsKey(bigramOcc + 1) &&
							goodTuringFreq.containsKey(bigramOcc))
						goodTuringSmoothing = (((bigramOcc + 1.0) * goodTuringFreq.get(bigramOcc + 1)) /
								(goodTuringFreq.get(bigramOcc) * totNumOfWordsInVoc));
					else if (!goodTuringFreq.containsKey(bigramOcc + 1.0) &&
							goodTuringFreq.containsKey(bigramOcc))
						goodTuringSmoothing = ((bigramOcc + 1.0) / (goodTuringFreq.get(bigramOcc)*totNumOfWordsInVoc));
					else
						goodTuringSmoothing = ((bigramOcc + 1.0) / totNumOfWordsInVoc);
				}
				predictingBigrams.add(new AbstractMap.SimpleEntry(bigram, goodTuringSmoothing));
			}
		}
		predictNextWord();
	}

	//Main Function
	public static void main(String args[])
	{
		
		//Check args and take the Corpus file
		String sentence = null;
		boolean toContinue = true;
		wordPred biGrams = new wordPred();
		String fileName = args[0];

		biGrams.readCorpusFile(fileName);
		biGrams.computeBigrams();
		//Enter word or set of words
		System.out.println("Enter your line: \n");
		Scanner sc = new Scanner(System.in);
		sentence = sc.nextLine();
		boolean isInCorrect = true;
		
		while(toContinue){
			while(isInCorrect){
				sentence = sentence.replaceAll("[^\\w\\s]","");
				if(sentence.toLowerCase().equals("null") || sentence == null ||  sentence.equals("") ){
					System.out.println("Input can't contain special characters only or can't be empty or null!");
					System.out.println("Enter the input here again: ");
					sentence = sc.nextLine();
					continue;
				} else break;
			}
			
			//Ask user to set a word or a set of words
			//Predict and display the words
			//Ask user if he wishes to continue
			sentence = sentence.toLowerCase();
			System.out.println("Your line is: " + sentence.substring(0,1).toUpperCase()+sentence.substring(1,sentence.length()));
			System.out.println("Predicting next word..");
			biGrams.computeBigramModelWithGoodTuringSmoothing(sentence);
			System.out.println("\nEnter your choice of word: ");
			String choice = sc.nextLine();
			sentence = sentence.concat(" ").concat(choice.toLowerCase());

			System.out.println("\n");
			System.out.println("Do you wish to continue? (Yes/No)");
			String answer = sc.nextLine();
			if(answer.equalsIgnoreCase("Yes") || answer.equalsIgnoreCase("Y"))
				toContinue = true;
			else if(answer.equalsIgnoreCase("No") || answer.equalsIgnoreCase("N"))
				toContinue = false;
			else{
				boolean invalidAnswer = true;
				while(invalidAnswer){
					System.out.println("Invalid choice! Please choose again!");
					answer = sc.nextLine();
					if(answer.equalsIgnoreCase("Yes") || answer.equalsIgnoreCase("Y")){
						toContinue = true;
						invalidAnswer = false;
					}else if(answer.equalsIgnoreCase("No") || answer.equalsIgnoreCase("N")){
						break;
					}
				}
			}
		}
	}
}
