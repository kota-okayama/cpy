"""GPT Core"""

from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import os
import random
import re
import numpy as np
import multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
import time
from datetime import datetime, timedelta
import json
import hashlib
import matplotlib.pyplot as plt
import requests

from .utils import path
from .logger import Logger, LoggerConfig
from .file_container import RecordContainer
from .datatype.workflow import WorkflowConfig, CacheMode, WorkflowState
from .datatype.record import RecordType, RecordMG

# Constants - keeping them similar to FasttextCore
EPOCHS = 30
MARGIN = 10
THRESHOLD = 5

class GPTCore:
    """GPT-based entity matching class to replace FasttextCore"""

    def __init__(
        self,
        config: WorkflowConfig = None,
        target_filepath: str = None,
        log_filepath: str = None,
        inf_attr: Dict[str, RecordType] = {},
        api_key: str = None,
        model: str = "gpt-4o",
    ):
        """Constructor for GPTCore"""
        self.inf_attr = inf_attr  # Attributes used for inference
        self.config = config
        self.target_filepath = target_filepath
        self.log_filepath = log_filepath
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        
        # Initialize logger
        self.logger: Logger = Logger(
            __name__,
            logger_config=LoggerConfig(level=self.config.log_level if config else "INFO"),
            filepath=self.log_filepath,
        )
        
        # Embedding cache to avoid redundant API calls
        self.embedding_cache = {}
        
        # Similarity model cache
        self.similarity_model = None
        
        # Threshold for matching determination
        self.similarity_threshold = 0.85  # Default threshold, can be tuned

    def gpt_api_call(self, messages: List[Dict[str, str]], temperature: float = 0) -> Dict:
        """Make an API call to OpenAI GPT"""
        if not self.api_key:
            raise ValueError("OpenAI API key is not set. Set it in the constructor or as OPENAI_API_KEY env variable.")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code != 200:
            self.logger.error(f"API call failed: {response.text}")
            raise Exception(f"API call failed: {response.text}")
            
        return response.json()

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embeddings for a text string using OpenAI's embedding API"""
        # Check cache first to avoid redundant API calls
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "text-embedding-3-large",
            "input": text
        }
        
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code != 200:
            self.logger.error(f"Embedding API call failed: {response.text}")
            raise Exception(f"Embedding API call failed: {response.text}")
        
        embedding = np.array(response.json()["data"][0]["embedding"])
        
        # Cache the embedding
        self.embedding_cache[text] = embedding
        
        return embedding

    def construct_vector(self, record: RecordMG, counter = None) -> np.ndarray:
        """
        Construct a vector representation for a record using GPT embeddings
        
        Parameters
        ----------
        record: RecordMG
            The record
        counter: Synchronized[int], optional
            Counter for parallel processing
        
        Returns
        -------
        np.ndarray
            Vector representation of the record
        """
        # Concatenate relevant record attributes
        text_to_embed = ""
        
        for attr, record_type in self.inf_attr.items():
            # Similar to FasttextCore's weighting strategy
            if record_type == RecordType.TEXT_KEY:
                text_to_embed += record.re.data[attr] + " " + record.re.data[attr] + " " + record.re.data[attr] + " "
            elif record_type == RecordType.COMPLEMENT_DATE:
                # For dates, use a special format to help the model understand
                text_to_embed += f"DATE: {record.re.data[attr]} "
            else:
                text_to_embed += record.re.data[attr] + " "
        
        # Get embedding from API
        embedding = self.get_embedding(text_to_embed.strip())
        
        # Increment counter if provided
        if counter is not None:
            counter.value += 1
            
        return embedding

    def construct_string(self, record: RecordMG) -> str:
        """
        Construct a concatenated string from a record
        
        Parameters
        ----------
        record: RecordMG
            The record
        
        Returns
        -------
        str
            Concatenated string representation of the record
        """
        result = ""
        for attr, record_type in self.inf_attr.items():
            # For dates, hash them similarly to FasttextCore
            if record_type == RecordType.COMPLEMENT_DATE:
                result += self.hash_number_for_date(record.re.data[attr])
            else:
                result += record.re.data[attr]
                
        return result

    def hash_number_for_date(self, text: str, str_slice: int = 0) -> str:
        """
        Hash a date string, similar to FasttextCore
        
        Parameters
        ----------
        text: str
            Date string to hash
        str_slice: int, optional
            Length limit for the hash
            
        Returns
        -------
        str
            Hashed date string
        """
        # Handle empty strings
        if text == "":
            text = "{}".format(random.randrange(300000, 1000000))
            
        # Preprocess dates
        date_list: List[str] = text.split(".")
        if len(date_list) == 2:
            date_list[1] = ("0" + date_list[1])[-2:]
        date = "".join(date_list)
            
        # Get hash of numbers only
        result = re.sub(r"\D", "", hashlib.sha256(date.encode()).hexdigest())
            
        if str_slice != 0:
            result = result[:str_slice]
            
        return result

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0
            
        return dot_product / (norm_a * norm_b)

    def euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate euclidean distance between two vectors"""
        return np.linalg.norm(vec1 - vec2)

    def train(
        self,
        filepath: str,
        test_ratio: float = 0.1,
        data_shuffle: bool = False,
        match_num: Optional[int] = None,
        mismatch_ratio: float = 1,
        max_in_cluster: int = 50,
        basemodel_filepath: Optional[str] = None,
        image_dirpath: str = None,
        use_crowdsourcing: bool = True,
        use_best_model: bool = True,
    ) -> WorkflowState:
        """
        Train a GPT-based entity matching model
        
        Parameters
        ----------
        filepath: str
            Path to training data
        test_ratio: float
            Ratio of data to use for testing
        data_shuffle: bool
            Whether to shuffle data
        match_num: int, optional
            Number of matching pairs to include
        mismatch_ratio: float
            Ratio of mismatched pairs to matching pairs
        max_in_cluster: int
            Maximum number of records in a cluster
        basemodel_filepath: str, optional
            Path to base model
        image_dirpath: str
            Directory to save visualization images
        use_crowdsourcing: bool
            Whether to use crowdsourcing results
        use_best_model: bool
            Whether to use the best model
            
        Returns
        -------
        WorkflowState
            State of the workflow
        """
        state = WorkflowState()
        self.logger.info("Starting GPT-based entity matching training")
        
        # Create the image directory if it doesn't exist
        if image_dirpath and not path.exists(image_dirpath):
            os.makedirs(image_dirpath)
        
        # Load training data
        rc = RecordContainer(log_filepath=self.log_filepath)
        rc.load_file(filepath)
        rc_recordmg = rc.get_recordmg()
        
        # Get matching and mismatching pairs
        (
            match_pairs,
            mismatch_pairs,
            match_labels,
            mismatch_labels,
        ) = rc.get_recordmg_for_train(
            match_num=match_num,
            mismatch_ratio=mismatch_ratio,
            max_in_cluster=max_in_cluster,
            labeling_function=self.difflib_driven_labeling,
        )
        
        # Process pairs to get vectors
        self.logger.info(f"Processing {len(match_pairs)//2} matching pairs and {len(mismatch_pairs)//2} mismatching pairs")
        
        # Get vectors for all records
        record_array = self.get_vectors_with_mp(
            CacheMode.WRITE, 
            filepath, 
            rc_recordmg
        )
        
        # Organize matching pairs
        match_vectors = []
        for i in range(len(match_pairs) // 2):
            match_vectors.append([
                record_array[match_pairs[i * 2].re.idx],
                record_array[match_pairs[i * 2 + 1].re.idx]
            ])
        
        # Organize mismatching pairs
        mismatch_vectors = []
        for i in range(len(mismatch_pairs) // 2):
            mismatch_vectors.append([
                record_array[mismatch_pairs[i * 2].re.idx],
                record_array[mismatch_pairs[i * 2 + 1].re.idx]
            ])
        
        # Add crowdsourcing results if enabled and available
        if use_crowdsourcing and self.config and len(self.config.crowdsourcing_result) > 0:
            self.logger.info("Including crowdsourcing results in training")
            
            # Load crowdsourcing pairs
            (
                cs_match_pairs,
                cs_mismatch_pairs,
                cs_match_labels,
                cs_mismatch_labels,
            ) = self.load_crowdsourcing_pair(labeling_function=self.difflib_driven_labeling)
            
            # Process crowdsourcing data
            cs_rc = RecordContainer(log_filepath=self.log_filepath)
            cs_rc.load_file(self.target_filepath)
            cs_rc_recordmg = cs_rc.get_recordmg()
            
            cs_record_array = self.get_vectors_with_mp(
                CacheMode.WRITE,
                self.target_filepath,
                cs_rc_recordmg
            )
            
            # Add crowdsourcing match pairs
            cs_match_vectors = []
            for i in range(len(cs_match_pairs) // 2):
                cs_match_vectors.append([
                    cs_record_array[cs_match_pairs[i * 2].re.idx],
                    cs_record_array[cs_match_pairs[i * 2 + 1].re.idx]
                ])
            
            # Add crowdsourcing mismatch pairs
            cs_mismatch_vectors = []
            for i in range(len(cs_mismatch_pairs) // 2):
                cs_mismatch_vectors.append([
                    cs_record_array[cs_mismatch_pairs[i * 2].re.idx],
                    cs_record_array[cs_mismatch_pairs[i * 2 + 1].re.idx]
                ])
            
            # Combine with existing vectors
            match_vectors.extend(cs_match_vectors)
            mismatch_vectors.extend(cs_mismatch_vectors)
            match_labels.extend(cs_match_labels)
            mismatch_labels.extend(cs_mismatch_labels)
        
        # Convert to numpy arrays
        match_vectors = np.array(match_vectors)
        mismatch_vectors = np.array(mismatch_vectors)
        match_labels = np.array(match_labels)
        mismatch_labels = np.array(mismatch_labels)
        
        # Shuffle if required
        if data_shuffle:
            match_indices = np.random.permutation(len(match_vectors))
            match_vectors = match_vectors[match_indices]
            match_labels = match_labels[match_indices]
            
            mismatch_indices = np.random.permutation(len(mismatch_vectors))
            mismatch_vectors = mismatch_vectors[mismatch_indices]
            mismatch_labels = mismatch_labels[mismatch_indices]
        
        # Split into training and testing sets
        match_train_count = int(len(match_vectors) * (1 - test_ratio))
        mismatch_train_count = int(len(mismatch_vectors) * (1 - test_ratio))
        
        train_vectors = np.concatenate([
            match_vectors[:match_train_count],
            mismatch_vectors[:mismatch_train_count]
        ])
        train_labels = np.concatenate([
            match_labels[:match_train_count],
            mismatch_labels[:mismatch_train_count]
        ])
        
        test_vectors = np.concatenate([
            match_vectors[match_train_count:],
            mismatch_vectors[mismatch_train_count:]
        ])
        test_labels = np.concatenate([
            match_labels[match_train_count:],
            mismatch_labels[mismatch_train_count:]
        ])
        
        # Train the model using cosine similarity optimization
        self.logger.info("Training similarity model")
        
        # Instead of neural network training, we'll tune the threshold for similarity
        # based on the training data
        similarities = []
        for pair in train_vectors:
            similarity = self.cosine_similarity(pair[0], pair[1])
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        true_match = train_labels > 0.5  # Convert to boolean mask
        
        # Calculate metrics for different thresholds
        thresholds = np.arange(0.5, 1.0, 0.01)
        best_f1 = 0
        best_threshold = 0.85  # Default if training fails
        
        for threshold in thresholds:
            predicted_match = similarities >= threshold
            
            # Calculate precision, recall, F1
            true_positives = np.sum(predicted_match & true_match)
            false_positives = np.sum(predicted_match & ~true_match)
            false_negatives = np.sum(~predicted_match & true_match)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.similarity_threshold = best_threshold
        self.logger.info(f"Optimized similarity threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
        
        # Evaluate on test set
        test_similarities = []
        for pair in test_vectors:
            similarity = self.cosine_similarity(pair[0], pair[1])
            test_similarities.append(similarity)
        
        test_similarities = np.array(test_similarities)
        test_true_match = test_labels > 0.5
        test_predicted_match = test_similarities >= best_threshold
        
        test_accuracy = np.mean(test_predicted_match == test_true_match)
        self.logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Create visualization if image directory is specified
        if image_dirpath:
            self.logger.info("Creating visualizations")
            
            # Find an available filename
            loss_filepath = None
            for i in range(1, 10000):
                tmpname = path.join(image_dirpath, f"similarity-{i:04d}.png")
                if not path.exists(tmpname):
                    loss_filepath = tmpname
                    break
            
            # Create similarity distribution plot
            plt.figure(figsize=(10, 6))
            plt.hist(
                [test_similarities[test_true_match], test_similarities[~test_true_match]],
                bins=30,
                alpha=0.7,
                label=['Match', 'Mismatch']
            )
            plt.axvline(best_threshold, color='red', linestyle='dashed', linewidth=2)
            plt.title('Similarity Distribution (Test Set)')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(True)
            plt.savefig(loss_filepath)
            plt.close()
            
            # Create distribution comparison for training/test
            acc_filepath = None
            for i in range(1, 10000):
                tmpname = path.join(image_dirpath, f"accuracy-{i:04d}.png")
                if not path.exists(tmpname):
                    acc_filepath = tmpname
                    break
            
            plt.figure(figsize=(10, 6))
            plt.scatter(
                np.arange(len(test_similarities)),
                test_similarities,
                c=test_true_match,
                alpha=0.6,
                cmap='coolwarm'
            )
            plt.axhline(best_threshold, color='black', linestyle='dashed')
            plt.title('Similarity vs Ground Truth (Test Set)')
            plt.xlabel('Sample Index')
            plt.ylabel('Cosine Similarity')
            plt.colorbar(label='Is Match')
            plt.grid(True)
            plt.savefig(acc_filepath)
            plt.close()
        
        self.logger.info("GPT-based entity matching training completed")
        state.finished = True
        return state

    def load_crowdsourcing_pair(
        self,
        labeling_function: Callable[[RecordMG, RecordMG, int], float] = None,
    ) -> Tuple[List[RecordMG], List[RecordMG], List[float], List[float]]:
        """
        Load and process crowdsourcing pairs
        
        Parameters
        ----------
        labeling_function: Callable, optional
            Function to determine labels
            
        Returns
        -------
        Tuple
            Matching pairs, mismatching pairs, matching labels, mismatching labels
        """
        rc = RecordContainer(log_filepath=self.log_filepath)
        rc.load_file(self.target_filepath)
        
        match_pairs = []
        match_labels = []
        mismatch_pairs = []
        mismatch_labels = []
        
        for key, value in self.config.crowdsourcing_result.items():
            if sum(value) / len(value) >= 0.5:
                match_pairs.append(RecordMG(rc.records[key[0]], self.inf_attr))
                match_pairs.append(RecordMG(rc.records[key[1]], self.inf_attr))
                
                if labeling_function is not None:
                    match_labels.append(labeling_function(match_pairs[-2], match_pairs[-1], 1))
                else:
                    match_labels.append(1)
            else:
                mismatch_pairs.append(RecordMG(rc.records[key[0]], self.inf_attr))
                mismatch_pairs.append(RecordMG(rc.records[key[1]], self.inf_attr))
                
                if labeling_function is not None:
                    mismatch_labels.append(labeling_function(mismatch_pairs[-2], mismatch_pairs[-1], 0))
                else:
                    mismatch_labels.append(0)
                    
        return (match_pairs, mismatch_pairs, match_labels, mismatch_labels)
    
    def difflib_driven_labeling(self, recordmg1: RecordMG, recordmg2: RecordMG, label: int) -> float:
        """
        Generate a label based on similarity between records
        
        Parameters
        ----------
        recordmg1: RecordMG
            First record
        recordmg2: RecordMG
            Second record
        label: int
            Expected label (1 for match, 0 for mismatch)
            
        Returns
        -------
        float
            Generated label value
        """
        from difflib import SequenceMatcher
        
        # Get string representations of records
        str1 = self.construct_string(recordmg1)
        str2 = self.construct_string(recordmg2)
        
        # Get similarity ratio
        similarity = SequenceMatcher(None, str1, str2).ratio()
        
        if label == 1:
            # For matches, return 1 minus similarity
            return 1 - similarity
        else:
            # For mismatches, return 1 minus similarity, scaled and offset
            return (1 - similarity) * (MARGIN - 1) + 1

    def load_model(self, filepath: str) -> WorkflowState:
        """
        Load a previously saved model
        
        Parameters
        ----------
        filepath: str
            Path to saved model
            
        Returns
        -------
        WorkflowState
            State of the workflow
        """
        state = WorkflowState()
        
        self.logger.info(f"Loading model from {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
                
            self.similarity_threshold = model_data.get('similarity_threshold', 0.85)
            self.logger.info(f"Loaded model with similarity threshold: {self.similarity_threshold}")
            
            state.finished = True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            state.finished = False
            
        return state

    def save_model(self, filepath: str) -> WorkflowState:
        """
        Save the current model
        
        Parameters
        ----------
        filepath: str
            Path to save model
            
        Returns
        -------
        WorkflowState
            State of the workflow
        """
        state = WorkflowState()
        
        self.logger.info(f"Saving model to {filepath}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            # Save only essential parameters
            model_data = {
                'similarity_threshold': self.similarity_threshold,
            }
            
            with open(filepath, 'w') as f:
                json.dump(model_data, f)
                
            self.logger.info(f"Model saved to {filepath}")
            state.finished = True
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            state.finished = False
            
        return state

    def match_pair_accuracy(self, image_dirpath: str = None) -> WorkflowState:
        """
        Calculate and visualize accuracy of matching pairs
        
        Parameters
        ----------
        image_dirpath: str, optional
            Directory to save visualization images
            
        Returns
        -------
        WorkflowState
            State of the workflow
        """
        state = WorkflowState()
        
        # Create directory if it doesn't exist
        if image_dirpath and not path.exists(image_dirpath):
            os.makedirs(image_dirpath)
            
        # Get matching pairs from target file
        rc = RecordContainer(log_filepath=self.log_filepath)
        rc.load_file(self.target_filepath)
        idx_pairs = rc.get_all_match_pairs_index()
        
        # Get vectors for all records
        record_array = self.get_vectors_with_mp(CacheMode.WRITE, self.target_filepath)
        
        # Calculate similarities for all pairs
        similarities = []
        for a, b in idx_pairs:
            similarity = self.cosine_similarity(record_array[a], record_array[b])
            similarities.append(similarity)
            
        # Create visualization
        if image_dirpath:
            # Find an available filename
            filepath = None
            for i in range(1, 10000):
                tmpname = path.join(image_dirpath, f"matchpairs-{i:04d}.png")
                if not path.exists(tmpname):
                    filepath = tmpname
                    break
                    
            plt.figure(figsize=(10, 6))
            plt.title("Similarity of Match Pairs")
            plt.plot(
                np.arange(len(similarities)),
                similarities,
                linestyle="none",
                marker="o",
                markersize=5,
                alpha=0.3
            )
            plt.axhline(self.similarity_threshold, color='red', linestyle='dashed')
            plt.xlabel("Pair Index")
            plt.ylabel("Cosine Similarity")
            plt.grid(True)
            plt.savefig(filepath)
            plt.close()
            
        # Calculate statistics
        similarities = np.array(similarities)
        avg_similarity = np.mean(similarities)
        median_similarity = np.median(similarities)
        min_similarity = np.min(similarities)
        max_similarity = np.max(similarities)
        
        self.logger.info(f"Match pair statistics:")
        self.logger.info(f"  Count: {len(similarities)}")
        self.logger.info(f"  Average similarity: {avg_similarity:.4f}")
        self.logger.info(f"  Median similarity: {median_similarity:.4f}")
        self.logger.info(f"  Min similarity: {min_similarity:.4f}")
        self.logger.info(f"  Max similarity: {max_similarity:.4f}")
        self.logger.info(f"  Pairs below threshold ({self.similarity_threshold}): {np.sum(similarities < self.similarity_threshold)}")
        
        state.finished = True
        return state

    def get_distance(self, pairs: np.ndarray) -> np.ndarray:
        """
        Calculate distances between pairs of vectors
        
        Parameters
        ----------
        pairs: np.ndarray
            Array of vector pairs
            
        Returns
        -------
        np.ndarray
            Array of distances
        """
        if pairs.ndim == 3:  # Batch of pairs
            distances = []
            for pair in pairs:
                # Convert cosine similarity to distance (1 - similarity)
                similarity = self.cosine_similarity(pair[0], pair[1])
                distance = 1 - similarity
                # Scale to match the original scale in FasttextCore
                scaled_distance = distance * 10
                distances.append(scaled_distance)
            return np.array(distances)
        else:  # Single pair
            similarity = self.cosine_similarity(pairs[0], pairs[1])
            distance = 1 - similarity
            scaled_distance = distance * 10
            return np.array([scaled_distance])

    def predict(self, vectors: np.ndarray) -> np.ndarray:
        """
        Transform vectors using the trained model
        
        Parameters
        ----------
        vectors: np.ndarray
            Input vectors
            
        Returns
        -------
        np.ndarray
            Transformed vectors
        """
        # In our GPT-based approach, we return the vectors as-is
        # since we're relying on cosine similarity directly
        return vectors

    def get_vectors_with_mp(
        self,
        cache_mode: CacheMode,
        filepath: str = None,
        records: List[RecordMG] = None,
        max_cores: int = None,
        log_minutes: int = 5
    ) -> np.ndarray:
        """
        Get vectors for records using multiprocessing
        
        Parameters
        ----------
        cache_mode: CacheMode
            How to handle caching
        filepath: str, optional
            Path to file containing records
        records: List[RecordMG], optional
            List of records
        max_cores: int, optional
            Maximum number of cores to use
        log_minutes: int, optional
            How often to log progress
            
        Returns
        -------
        np.ndarray
            Array of vectors
        """
        # Validate inputs
        if filepath is None and records is None:
            raise ValueError("Either filepath or records must be specified")
            
        # If filepath is not specified, set cache mode to NONE
        cache_mode = CacheMode.NONE if filepath is None else cache_mode
        
        cache_path = None
        
        # Check for cached vectors
        if cache_mode != CacheMode.NONE:
            cache_path = path.join(
                path.dirname(filepath),
                ".cache",
                f"{path.splitext(path.basename(filepath))[0]}.gpt.npy"
            )
            
            if path.exists(cache_path):
                try:
                    self.logger.info(f"Loading vectors from cache: {cache_path}")
                    return np.load(cache_path)
                except Exception as e:
                    self.logger.error(f"Failed to load cache: {e}")
        
        # Load records if not provided
        if records is None:
            rc = RecordContainer(log_filepath=self.log_filepath)
            rc.load_file(filepath)
            records = rc.get_recordmg()
            
        # Determine number of cores for parallel processing
        if max_cores is None:
            max_cores = max(1, mp.cpu_count() - 2)  # Reserve cores for system and responsiveness
        max_cores = max(1, min(max_cores, mp.cpu_count(), int(len(records) / 100)))
            
        # Process records in parallel to get vectors
        self.logger.info(f"Generating vectors for {len(records)} records using {max_cores} cores")
        call_time = datetime.now() + timedelta(minutes=log_minutes)
        
        with mp.Manager() as manager:
            counter = manager.Value('i', 0)
            with mp.Pool(processes=max_cores) as pool:
                async_result = pool.starmap_async(
                    self.construct_vector,
                    [(r, counter) for r in records]
                )
                
                # Monitor progress
                while not async_result.ready():
                    if datetime.now() > call_time:
                        total_records = len(records)
                        current_value = counter.value
                        progress = int(current_value * 100 / total_records)
                        self.logger.info(f"Progress: {current_value}/{total_records} ({progress}%)")
                        call_time += timedelta(minutes=log_minutes)
                    time.sleep(0.1)
                    
                record_vectors = async_result.get()
                
        # Convert to numpy array
        record_vectors = np.array(record_vectors)
        
        # Save cache if needed
        if cache_mode == CacheMode.WRITE and cache_path:
            # Ensure cache directory exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            # Save to cache
            np.save(cache_path, record_vectors)
            self.logger.info(f"Vectors saved to cache: {cache_path}")
            
        return record_vectors
        
    def get_similarity_recommendation(self, record1: RecordMG, record2: RecordMG) -> Dict[str, Any]:
        """
        Get detailed similarity recommendation between two records using GPT
        
        Parameters
        ----------
        record1: RecordMG
            First record
        record2: RecordMG
            Second record
            
        Returns
        -------
        Dict
            Dictionary containing similarity assessment and explanation
        """
        # Construct input for GPT
        record1_text = self.format_record_for_gpt(record1)
        record2_text = self.format_record_for_gpt(record2)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in entity matching and record deduplication. "
                    "You need to determine if two records refer to the same real-world entity. "
                    "Analyze the records carefully and provide a detailed explanation of your reasoning. "
                    "Consider all fields but give more weight to key identifiers like names and IDs. "
                    "Minor differences in spelling, formatting, or missing data should be handled intelligently."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Analyze these two records and determine if they represent the same entity:\n\n"
                    f"RECORD 1:\n{record1_text}\n\n"
                    f"RECORD 2:\n{record2_text}\n\n"
                    f"Are these records referring to the same entity? Respond with a JSON object containing:\n"
                    f"1. 'is_match': true or false\n"
                    f"2. 'confidence': a score from 0.0 to 1.0\n"
                    f"3. 'explanation': your detailed reasoning\n"
                    f"4. 'field_matches': an object showing similarity assessment for each field"
                )
            }
        ]
        
        try:
            # Make API call
            response = self.gpt_api_call(messages)
            content = response['choices'][0]['message']['content']
            
            # Parse JSON response
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # If GPT didn't return valid JSON, extract assessment manually
                is_match = "true" in content.lower() and "same entity" in content.lower()
                return {
                    "is_match": is_match,
                    "confidence": 0.7 if is_match else 0.3,  # Default confidence
                    "explanation": content,
                    "field_matches": {}
                }
                
        except Exception as e:
            self.logger.error(f"Error getting similarity recommendation: {e}")
            # Return default assessment based on embedding similarity
            vec1 = self.construct_vector(record1)
            vec2 = self.construct_vector(record2)
            similarity = self.cosine_similarity(vec1, vec2)
            
            return {
                "is_match": similarity >= self.similarity_threshold,
                "confidence": similarity,
                "explanation": f"Automated assessment based on embedding similarity ({similarity:.4f})",
                "field_matches": {}
            }
    
    def format_record_for_gpt(self, record: RecordMG) -> str:
        """
        Format a record for GPT input
        
        Parameters
        ----------
        record: RecordMG
            Record to format
            
        Returns
        -------
        str
            Formatted record
        """
        result = ""
        for attr, record_type in self.inf_attr.items():
            value = record.re.data.get(attr, "")
            result += f"{attr}: {value}\n"
            
        return result
    
    def batch_entity_resolution(
        self,
        records: List[RecordMG],
        threshold: float = None,
        max_pairs: int = 1000,
        detailed: bool = False
    ) -> List[Tuple[int, int, float, Dict]]:
        """
        Perform batch entity resolution on a list of records
        
        Parameters
        ----------
        records: List[RecordMG]
            List of records
        threshold: float, optional
            Similarity threshold (uses model threshold if not specified)
        max_pairs: int, optional
            Maximum number of pairs to analyze in detail
        detailed: bool, optional
            Whether to get detailed recommendations for top pairs
            
        Returns
        -------
        List[Tuple]
            List of matching pairs with similarity scores
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        self.logger.info(f"Starting batch entity resolution for {len(records)} records")
        
        # Get vectors for all records
        record_vectors = np.array([self.construct_vector(r) for r in records])
        
        # Calculate pairwise similarities
        pairs = []
        for i in range(len(records)):
            for j in range(i+1, len(records)):
                similarity = self.cosine_similarity(record_vectors[i], record_vectors[j])
                if similarity >= threshold:
                    pairs.append((i, j, similarity, {}))
        
        # Sort by similarity (descending)
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Get detailed recommendations for top pairs if requested
        if detailed and pairs:
            top_pairs = pairs[:min(max_pairs, len(pairs))]
            self.logger.info(f"Getting detailed recommendations for {len(top_pairs)} pairs")
            
            for idx, (i, j, similarity, _) in enumerate(top_pairs):
                try:
                    recommendation = self.get_similarity_recommendation(records[i], records[j])
                    pairs[idx] = (i, j, similarity, recommendation)
                except Exception as e:
                    self.logger.error(f"Error getting recommendation for pair ({i},{j}): {e}")
                    
        self.logger.info(f"Found {len(pairs)} potential matches")
        return pairs