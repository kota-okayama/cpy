import yaml
import re
import json
import unicodedata
import asyncio
import aiohttp
import os
import uuid
import time
import math
import copy
import argparse
import random
import argparse
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from jellyfish import jaro_winkler_similarity

import yaml
import re
import json
import unicodedata
import asyncio
import aiohttp
import os
import uuid
import time
import math
import copy
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from jellyfish import jaro_winkler_similarity

# Union-Find data structure for cluster management
class UnionFind:
    """Union-Find data structure implementation for cluster management"""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        """Find the root of x (with path compression)"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Merge x and y (with rank optimization)"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        else:
            self.parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1

# Text normalization function
def normalize_text(text: str) -> str:
    """Normalize text by removing special characters and standardizing format"""
    if not isinstance(text, str):
        return ""
    # Normalize Unicode
    text = unicodedata.normalize('NFKC', text)
    # Remove brackets and special characters
    text = re.sub(r'[\[\]\(\)\{\}【】〔〕]', ' ', text)
    # Replace consecutive spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Extract publication year
def extract_year(pubdate: str) -> Optional[str]:
    """Extract year from publication date"""
    if not pubdate:
        return None
    # Search for 4-digit number pattern
    year_match = re.search(r'(19|20)\d{2}', pubdate)
    if year_match:
        return year_match.group(0)
    return None

# Extract representatives from YAML data
def extract_representatives(yaml_data: str) -> List[Dict[str, Any]]:
    """Extract representative records from each cluster in YAML data"""
    try:
        data = yaml.safe_load(yaml_data)
        representatives = []
        
        # Process dictionary format
        if isinstance(data, dict):
            if 'group' in data and isinstance(data['group'], list):
                print(f"Found group list ({len(data['group'])} groups)")
                
                for group_idx, group in enumerate(data['group']):
                    if isinstance(group, dict) and 'records' in group:
                        if group['records']:  # Only process if records exist
                            print(f"Group {group_idx} record count: {len(group['records'])}")
                            rep = get_cluster_representative(group['records'])
                            if rep:
                                rep_data = rep.get('data', {})
                                representatives.append({
                                    "cluster_id": group_idx,
                                    "title": rep_data.get('bib1_title', ''),
                                    "author": rep_data.get('bib1_author', ''),
                                    "publisher": rep_data.get('bib1_publisher', ''),
                                    "pubdate": rep_data.get('bib1_pubdate', ''),
                                    "original_id": rep.get('id', ''),
                                    "original_idx": rep.get('idx', -1),
                                    "original_cluster_id": group_idx,
                                    "all_records": group['records']
                                })
        
        print(f"Extracted {len(representatives)} representative records")
        return representatives
    
    except Exception as e:
        print(f"Error during representative extraction: {e}")
        import traceback
        traceback.print_exc()
        return []

# Select cluster representative
def get_cluster_representative(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Select the most informative record as the cluster representative"""
    if not records:
        return None
    
    try:
        best_record = None
        max_fields = -1
        
        for record in records:
            # Create data field if it doesn't exist
            if 'data' not in record:
                if all(k in record for k in ['bib1_title', 'bib1_author', 'bib1_publisher', 'bib1_pubdate']):
                    record_data = {
                        'bib1_title': record.get('bib1_title', ''),
                        'bib1_author': record.get('bib1_author', ''),
                        'bib1_publisher': record.get('bib1_publisher', ''),
                        'bib1_pubdate': record.get('bib1_pubdate', '')
                    }
                    record['data'] = record_data
            
            data = record.get('data', {})
            
            # Count non-empty fields
            non_empty_fields = sum(1 for v in data.values() if v and isinstance(v, str) and len(v.strip()) > 0)
            
            # Increase priority for records with title and author
            has_title = bool(data.get('bib1_title', '').strip())
            has_author = bool(data.get('bib1_author', '').strip())
            priority_score = non_empty_fields + (2 if has_title else 0) + (1 if has_author else 0)
            
            if priority_score > max_fields:
                max_fields = priority_score
                best_record = record
        
        return best_record
    
    except Exception as e:
        print(f"Error selecting representative record: {e}")
        return records[0] if records else None

# Extract all records from input data
def extract_records(data: Any) -> List[Dict[str, Any]]:
    """Extract all records from input data"""
    records = []
    
    try:
        if isinstance(data, dict):
            if 'group' in data and isinstance(data['group'], list):
                for group_idx, group in enumerate(data['group']):
                    if isinstance(group, dict) and 'records' in group and group['records']:
                        for record in group['records']:
                            record_copy = copy.deepcopy(record)
                            if 'original_cluster_id' not in record_copy:
                                record_copy['original_cluster_id'] = str(group_idx)
                            records.append(record_copy)
        
        elif isinstance(data, list):
            for group_idx, group in enumerate(data):
                if isinstance(group, dict) and 'records' in group:
                    for record in group['records']:
                        record_copy = copy.deepcopy(record)
                        if 'original_cluster_id' not in record_copy:
                            record_copy['original_cluster_id'] = str(group_idx)
                        records.append(record_copy)
        
        print(f"Extracted a total of {len(records)} records")
        return records
    
    except Exception as e:
        print(f"Error during record extraction: {e}")
        import traceback
        traceback.print_exc()
        return []

# Precompute features
def precompute_features(representatives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract and precompute features from representative records"""
    features = []
    
    for rep in representatives:
        # Normalize text
        title = normalize_text(rep['title'])
        author = normalize_text(rep['author'])
        publisher = normalize_text(rep['publisher'])
        
        # Extract main title part
        main_title_match = re.match(r'^([^\(\[\{【〔]+)', title)
        main_title = main_title_match.group(1).strip() if main_title_match else title
        
        # Extract series numbers or volume numbers
        volume_numbers = re.findall(r'\d+', title)
        
        # Extract publication year
        year = extract_year(rep['pubdate'])
        
        features.append({
            "cluster_id": rep['cluster_id'],
            "main_title": main_title,
            "full_title": title,
            "author": author,
            "publisher": publisher,
            "year": year,
            "volume_info": volume_numbers,
            "original_data": rep,
            "original_cluster_id": rep.get('original_cluster_id', '')
        })
    
    return features

# Create efficient API request data
def create_efficient_api_request(features_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create optimized request data for GPT API"""
    request_data = {
        "records": []
    }
    
    for feature in features_batch:
        record_data = {
            "id": feature['cluster_id'],
            "main_title": feature['main_title'],
            "full_title": feature['full_title'],
            "author": feature['author'],
            "publisher": feature['publisher'],
            "year": feature['year'],
            "volume_info": feature['volume_info'],
            "original_cluster_id": feature.get('original_cluster_id', '')
        }
        request_data["records"].append(record_data)
    
    return request_data

# Create GPT prompt
def create_gpt_prompt(batch_data: Dict[str, Any]) -> str:
    """Create prompt for GPT API"""
    records = batch_data["records"]
    
    prompt = """
You are an expert in analyzing similarity between bibliographic records. Calculate the similarity score (0-1) between each pair of book records.

Consider the following factors when calculating similarity:
- Title similarity (most important, weight: 60%)
- Author name similarity (weight: 30%)
- Publisher similarity (weight: 5%)
- Publication year similarity (weight: 5%)

Important:
- Different volumes of the same series have high similarity but are not complete matches (e.g. 0.8-0.9)
- Records with exactly matching title and author should have very high similarity (0.95-1.0)
- Japanese titles can have different notations, so normalize before comparison

Book records:
"""
    
    # Add record information
    for i, record in enumerate(records):
        prompt += f"\nRecord {i} (ID: {record['id']}):\n"
        prompt += f"  Main title: {record['main_title']}\n"
        prompt += f"  Full title: {record['full_title']}\n"
        prompt += f"  Author: {record['author']}\n"
        prompt += f"  Publisher: {record['publisher']}\n"
        prompt += f"  Publication year: {record['year'] if record['year'] else 'Unknown'}\n"
        prompt += f"  Volume info: {', '.join(record['volume_info']) if record['volume_info'] else 'Unknown'}\n"
    
    prompt += """
Instructions:
1. Calculate similarity between all possible record pairs
2. Only return pairs with similarity ≥ 0.5 (for more detailed analysis)
3. Return results in the following JSON format:

{
  "similarity_pairs": [
    {"pair": [0, 1], "score": 0.95, "reason": "Title and author match perfectly, likely the same series"},
    {"pair": [2, 4], "score": 0.85, "reason": "Appears to be different volumes of the same series"},
    ...
  ]
}

Return only JSON data. No explanations or additional text needed.
"""
    return prompt

# Call GPT API asynchronously
async def call_gpt_api_async(prompt: str, api_key: str) -> Dict[str, Any]:
    """Call GPT API asynchronously"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": "gpt-4o-mini",  # Adjust model as needed
            "messages": [
                {"role": "system", "content": "You are an expert in calculating similarity between Japanese bibliographic records."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,  # Lower temperature for more deterministic results
            "max_tokens": 2000,
            "response_format": {"type": "json_object"}  # Request JSON format response
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        async with session.post(
            "https://api.openai.com/v1/chat/completions", 
            json=payload, 
            headers=headers
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API call error: {response.status} - {error_text}")
            
            result = await response.json()
            return result

# Process requests in parallel
async def process_requests_parallel(batch_requests: List[Dict[str, Any]], api_key: str) -> List[Dict[str, Any]]:
    """Process multiple batch requests in parallel"""
    tasks = []
    for batch_data in batch_requests:
        prompt = create_gpt_prompt(batch_data)
        tasks.append(call_gpt_api_async(prompt, api_key))
    
    # Execute in parallel (with slight delay for rate limiting)
    results = []
    for i, task in enumerate(tasks):
        try:
            result = await task
            results.append({
                "batch_index": i,
                "batch_data": batch_requests[i],
                "api_response": result
            })
            # Wait a bit for rate limiting
            if i < len(tasks) - 1:
                await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            results.append({
                "batch_index": i,
                "batch_data": batch_requests[i],
                "error": str(e)
            })
    
    return results

# Extract similarity results
def extract_similarity_results(api_results: List[Dict[str, Any]], 
                           representatives: List[Dict[str, Any]],
                           output_dir: str = "results",
                           output_prefix: str = "result",
                           strategy: str = "core_inconsistency",
                           human_accuracy: float = 1.0) -> List[Dict[str, Any]]:
    """APIレスポンスから類似度結果を抽出して統合"""
    all_similarities = []
    
    # すべての判定結果を記録するリスト
    all_judgments = []
    
    for result in api_results:
        if "error" in result:
            print(f"エラーのためバッチ {result['batch_index']} の結果をスキップ: {result['error']}")
            continue
        
        try:
            # APIレスポンスから類似度ペアを抽出
            content = result["api_response"]["choices"][0]["message"]["content"]
            
            # JSONデータをサニタイズ
            sanitized_content = sanitize_json_response(content)
            
            # サニタイズされたJSONをパース
            response_data = json.loads(sanitized_content)
            
            if "similarity_pairs" not in response_data:
                print(f"バッチ {result['batch_index']} のレスポンスに 'similarity_pairs' がありません")
                continue
                        
            # バッチ内のレコードIDマッピング
            batch_records = result["batch_data"]["records"]
            id_mapping = {i: record["id"] for i, record in enumerate(batch_records)}
            
            # 類似度ペアを変換して追加
            for pair_data in response_data["similarity_pairs"]:
                local_ids = pair_data["pair"]
                global_ids = [id_mapping[local_id] for local_id in local_ids]
                
                # 元のレコード情報を取得
                record_info = []
                record_pairs = []
                for cluster_id in global_ids:
                    for rep in representatives:
                        if rep["cluster_id"] == cluster_id:
                            record_info.append({
                                "cluster_id": cluster_id,
                                "title": rep["title"],
                                "author": rep["author"],
                                "publisher": rep["publisher"],
                                "pubdate": rep["pubdate"],
                                "original_cluster_id": rep.get("original_cluster_id", "")
                            })
                            record_pairs.append(rep)
                            break
                
                # 類似度情報を追加
                similarity_entry = {
                    "cluster_pair": global_ids,
                    "similarity_score": pair_data["score"],
                    "reason": pair_data.get("reason", ""),
                    "records": record_info
                }
                all_similarities.append(similarity_entry)
                
                # 判定結果をすべて記録
                if len(record_pairs) == 2:
                    all_judgments.append({
                        "record1": {
                            "cluster_id": record_pairs[0]["cluster_id"],
                            "title": record_pairs[0]["title"],
                            "author": record_pairs[0]["author"],
                            "publisher": record_pairs[0]["publisher"],
                            "pubdate": record_pairs[0]["pubdate"]
                        },
                        "record2": {
                            "cluster_id": record_pairs[1]["cluster_id"],
                            "title": record_pairs[1]["title"],
                            "author": record_pairs[1]["author"],
                            "publisher": record_pairs[1]["publisher"],
                            "pubdate": record_pairs[1]["pubdate"]
                        },
                        "similarity_score": pair_data["score"],
                        "reason": pair_data.get("reason", ""),
                        "model": "gpt-4o-mini" # 初期モデル
                    })
        except Exception as e:
            print(f"バッチ {result['batch_index']} の結果処理中にエラーが発生: {e}")
    
    # 類似度スコアで降順ソート
    all_similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # すべての判定結果を保存
    judgments_file = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_initial_judgments.json"
    print(f"初期判定ファイルのパス: {judgments_file}")
    print(f"初期判定結果数: {len(all_judgments)}")
    
    try:
        # 出力ディレクトリが存在しない場合は作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        with open(judgments_file, 'w', encoding='utf-8') as f:
            json.dump(all_judgments, f, ensure_ascii=False, indent=2)
        print(f"初期類似度判定結果を {judgments_file} に保存しました（{len(all_judgments)}ペア）")
        
        # 判定エラーを分析して表示・保存
        analyze_judgment_errors(
            all_judgments, 
            representatives, 
            output_dir, 
            output_prefix, 
            strategy, 
            human_accuracy, 
            -1  # 初期判定の場合は-1
        )
        
    except Exception as e:
        print(f"ファイル保存中にエラーが発生: {e}")
    
    return all_similarities

# Sanitize JSON response
def sanitize_json_response(json_string: str) -> str:
    """Fix and sanitize JSON response to ensure it can be parsed"""
    if not json_string:
        return "{}"
    
    # 1. Handle unescaped quotes in strings
    cleaned = ""
    in_string = False
    escape_next = False
    
    for i, char in enumerate(json_string):
        if char == '"' and not escape_next:
            in_string = not in_string
            cleaned += char
        elif char == '\\':
            escape_next = True
            cleaned += char
        elif char == '"' and escape_next:
            escape_next = False
            cleaned += char
        elif in_string and char == '\n':
            # Remove line breaks in strings
            cleaned += " "
            escape_next = False
        else:
            cleaned += char
            escape_next = False
    
    # 2. Ensure strings are properly closed
    # Count quotes
    quote_count = cleaned.count('"')
    if quote_count % 2 != 0:
        # Add closing quote if odd number of quotes
        cleaned += '"'
    
    # 3. Fix commas - remove trailing commas and add missing ones
    lines = cleaned.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        line = line.rstrip()
        
        # Process non-final lines
        if i < len(lines) - 1:
            # Remove commas before closing brackets
            if line.rstrip().endswith(',') and lines[i+1].strip() in [']}', ']', '}']:
                line = line.rstrip(',')
            # Add commas after elements if missing
            elif not line.rstrip().endswith(',') and not line.rstrip().endswith('{') and \
                 not line.rstrip().endswith('[') and not line.rstrip().endswith('}') and \
                 not line.rstrip().endswith(']') and lines[i+1].strip() not in [']}', ']', '}']:
                line += ','
        
        fixed_lines.append(line)
    
    cleaned = '\n'.join(fixed_lines)
    
    # 4. Ensure property names are quoted
    import re
    property_pattern = r'([a-zA-Z0-9_]+):'
    cleaned = re.sub(property_pattern, r'"\1":', cleaned)
    
    try:
        # Try parsing the JSON to see if it's valid
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError as e:
        # If still invalid, return a fallback structure
        print(f"JSON still invalid after sanitization: {e}")
        error_info = {"error": str(e), "original_text_sample": json_string[:100] + "..."}
        
        # Match the expected response format
        if "similarity_results" in json_string:
            return json.dumps({"similarity_results": []})
        elif "similarity_pairs" in json_string:
            return json.dumps({"similarity_pairs": []})
        else:
            return json.dumps({"error": "Parse failed", "results": []})

# Create clusters from matching pairs
def create_clusters_from_matches(similarities: List[Dict[str, Any]], representatives: List[Dict[str, Any]], threshold: float = 0.7) -> tuple:
    """Create clusters from similarity results and merge entire clusters if representatives are similar"""
    
    # Initialize Union-Find data structure
    uf = UnionFind(len(representatives))
    
    # Map cluster IDs to indices
    cluster_id_to_index = {rep["cluster_id"]: i for i, rep in enumerate(representatives)}
    
    # Track merge decisions
    merge_decisions = []
    
    # Merge based on similarity
    merged_count = 0
    for similarity in similarities:
        actual_similarity = similarity["similarity_score"]
        if actual_similarity >= threshold:
            pair = similarity["cluster_pair"]
            if pair[0] in cluster_id_to_index and pair[1] in cluster_id_to_index:
                idx1 = cluster_id_to_index[pair[0]]
                idx2 = cluster_id_to_index[pair[1]]
                
                # Record cluster merge
                merge_info = {
                    "pair": [pair[0], pair[1]],
                    "titles": [representatives[idx1]['title'], representatives[idx2]['title']],
                    "similarity_score": similarity["similarity_score"],
                    "reason": similarity.get("reason", ""),
                    "merged": True
                }
                merge_decisions.append(merge_info)
                
                print(f"Merging clusters: {representatives[idx1]['title']} and {representatives[idx2]['title']} (similarity: {similarity['similarity_score']}, threshold: {threshold})")
                uf.union(idx1, idx2)
                merged_count += 1
                
        # Record non-merged pairs above a certain threshold
        elif similarity["similarity_score"] >= 0.5:
            pair = similarity["cluster_pair"]
            if pair[0] in cluster_id_to_index and pair[1] in cluster_id_to_index:
                idx1 = cluster_id_to_index[pair[0]]
                idx2 = cluster_id_to_index[pair[1]]
                
                # Record non-merged similarity pair
                not_merged_info = {
                    "pair": [pair[0], pair[1]],
                    "titles": [representatives[idx1]['title'], representatives[idx2]['title']],
                    "similarity_score": similarity["similarity_score"],
                    "reason": similarity.get("reason", ""),
                    "merged": False
                }
                merge_decisions.append(not_merged_info)
    
    print(f"Merged {merged_count} cluster pairs with threshold {threshold}")
    
    # Build clusters (including all records from merged clusters)
    merged_clusters = defaultdict(list)
    for i, rep in enumerate(representatives):
        root = uf.find(i)
        merged_clusters[root].append(rep)
    
    # Format results
    result_clusters = []
    for cluster_idx, members in merged_clusters.items():
        # Select representative (using first member for convenience)
        representative = members[0] if members else None
        
        # Collect all records from merged clusters
        all_merged_records = []
        for member in members:
            if "all_records" in member and member["all_records"]:
                all_merged_records.extend(member["all_records"])
        
        result_clusters.append({
            "cluster_id": str(uuid.uuid4()),
            "members": members,
            "representative": representative,
            "all_records": all_merged_records,
            "member_count": len(members),
            "record_count": len(all_merged_records)
        })
        
        print(f"Created merged cluster with {len(members)} merged clusters and {len(all_merged_records)} records")
    
    # Return both clusters and merge decisions
    return result_clusters, merge_decisions

# Format output groups
def format_output_groups(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format clusters into output groups"""
    output_groups = []
    
    print(f"Formatting output groups... Processing {len(clusters)} clusters")
    
    for cluster_idx, cluster in enumerate(clusters):
        records = []
        # Add debug info
        print(f"Processing cluster {cluster_idx}:")
        print(f"  Member count: {len(cluster.get('members', []))}")
        print(f"  Representative: {cluster.get('representative', {}).get('title', 'None')}")
        
        # Check all_records field
        if 'all_records' in cluster and cluster['all_records']:
            all_records = cluster['all_records']
            print(f"  Total record count: {len(all_records)}")
            
            for record in all_records:
                # Check if record has data field
                if isinstance(record, dict):
                    record_id = record.get('id', '')
                    
                    # Use data field directly if available
                    if 'data' in record and isinstance(record['data'], dict):
                        data_fields = record['data']
                    else:
                        # Look for necessary fields
                        data_fields = {}
                        for key, value in record.items():
                            if key.startswith('bib1_'):
                                data_fields[key] = value
                    
                    # Add record to output format
                    records.append({
                        "id": record_id,
                        "cluster_id": cluster.get("cluster_id", str(uuid.uuid4())),
                        "original_cluster_id": record.get("original_cluster_id", ""),
                        "data": {
                            "bib1_title": data_fields.get("bib1_title", ""),
                            "bib1_author": data_fields.get("bib1_author", ""),
                            "bib1_publisher": data_fields.get("bib1_publisher", ""),
                            "bib1_pubdate": data_fields.get("bib1_pubdate", "")
                        }
                    })
        
        # Calculate perfect match flag (all records have same original cluster ID)
        perfect_match = False
        if records:
            first_original_id = records[0].get("original_cluster_id", "")
            perfect_match = all(r.get("original_cluster_id", "") == first_original_id for r in records)
            print(f"  Output record count: {len(records)}, Perfect match: {perfect_match}")
        else:
            print("  Output record count: 0")
        
        group = {
            "correct": [[i] for i in range(len(records))],
            "perfect_match": perfect_match,
            "records": records
        }
        
        output_groups.append(group)
    
    print(f"Created {len(output_groups)} output groups")
    return output_groups

# Calculate output metrics
def calculate_output_metrics(groups: List[Dict], all_records: List[Dict]) -> Dict:
    """Calculate metrics for output summary"""
    total_records = sum(len(group.get("records", [])) for group in groups)
    
    # Count final clusters
    num_of_groups_inference = len(groups)
    
    # Count original clusters
    original_cluster_ids = set()
    for record in all_records:
        if "original_cluster_id" in record:
            original_cluster_ids.add(record["original_cluster_id"])
    
    num_of_groups_correct = len(original_cluster_ids)
    
    # Compare original and new clustering
    original_cluster_to_records = defaultdict(list)
    for record in all_records:
        if "id" in record and "original_cluster_id" in record:
            original_cluster_to_records[record["original_cluster_id"]].append(record["id"])
    
    new_cluster_to_records = defaultdict(list)
    for group in groups:
        for record in group.get("records", []):
            if "id" in record and "cluster_id" in record:
                new_cluster_to_records[record["cluster_id"]].append(record["id"])
    
    # Pair-level calculations
    original_pairs = set()
    for cluster_id, record_ids in original_cluster_to_records.items():
        for i in range(len(record_ids)):
            for j in range(i+1, len(record_ids)):
                pair = tuple(sorted([record_ids[i], record_ids[j]]))
                original_pairs.add(pair)
    
    new_pairs = set()
    for cluster_id, record_ids in new_cluster_to_records.items():
        for i in range(len(record_ids)):
            for j in range(i+1, len(record_ids)):
                pair = tuple(sorted([record_ids[i], record_ids[j]]))
                new_pairs.add(pair)
    
    # Correctly predicted pairs
    correct_pairs = original_pairs.intersection(new_pairs)
    
    # Pair-level precision and recall
    if len(new_pairs) > 0:
        precision_pair = len(correct_pairs) / len(new_pairs)
    else:
        precision_pair = 0
    
    if len(original_pairs) > 0:
        recall_pair = len(correct_pairs) / len(original_pairs)
    else:
        recall_pair = 0
    
    # F1 score calculation
    if precision_pair + recall_pair > 0:
        f1_pair = 2 * precision_pair * recall_pair / (precision_pair + recall_pair)
    else:
        f1_pair = 0
    
    # Group-level precision and recall
    precision_group_count = 0
    recall_group_count = 0
    total_new_records = 0
    total_original_records = 0
    
    # For each new cluster, find most overlapping original cluster
    for cluster_id, record_ids in new_cluster_to_records.items():
        max_overlap = 0
        for original_id, original_ids in original_cluster_to_records.items():
            overlap = len(set(record_ids).intersection(set(original_ids)))
            max_overlap = max(max_overlap, overlap)
        
        precision_group_count += max_overlap
        total_new_records += len(record_ids)
    
    # For each original cluster, find most overlapping new cluster
    for original_id, original_ids in original_cluster_to_records.items():
        max_overlap = 0
        for cluster_id, record_ids in new_cluster_to_records.items():
            overlap = len(set(original_ids).intersection(set(record_ids)))
            max_overlap = max(max_overlap, overlap)
        
        recall_group_count += max_overlap
        total_original_records += len(original_ids)
    
    # Group-level precision and recall
    if total_new_records > 0:
        precision_group = precision_group_count / total_new_records
    else:
        precision_group = 0
    
    if total_original_records > 0:
        recall_group = recall_group_count / total_original_records
    else:
        recall_group = 0
    
    # Calculate complete match groups
    complete_match_count = 0
    complete_matched_clusters = []
    
    for original_id, original_ids in original_cluster_to_records.items():
        original_set = set(original_ids)
        for cluster_id, record_ids in new_cluster_to_records.items():
            new_set = set(record_ids)
            if original_set == new_set:
                complete_match_count += 1
                complete_matched_clusters.append(original_id)
                break
    
    if len(original_cluster_to_records) > 0:
        complete_group = complete_match_count / len(original_cluster_to_records)
    else:
        complete_group = 0
    
    # Create metrics
    metrics = {
        "type": "RESULT",
        "num_of_record": total_records,
        "num_of_groups(correct)": num_of_groups_correct,
        "num_of_groups(inference)": num_of_groups_inference,
        "config_match": None,
        "config_mismatch": None,
        "crowdsourcing_count": len(all_records),
        "f1(pair)": f"{f1_pair:.5f}",
        "precision(pair)": f"{precision_pair:.5f}",
        "recall(pair)": f"{recall_pair:.5f}",
        "complete(group)": complete_group,
        "precision(group)": precision_group,
        "recall(group)": recall_group,
        "complete_group": str(complete_matched_clusters)
    }
    
    return metrics

# Format final output
def format_final_output(groups: List[Dict], metrics: Dict) -> Dict:
    """Format output in final format"""
    output = {
        "version": "3.1",
        "type": "RESULT",
        "id": str(uuid.uuid4()),
        "summary": metrics,
        "group": groups
    }
    
    return output

# Save result files
def save_result_files(results: Dict[str, Any], output_yaml: str, output_json: str = None) -> None:
    """Save results to YAML and JSON files"""
    # Create output directory
    output_dir = os.path.dirname(output_yaml)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory {output_dir}")
    
    # Save to YAML file
    with open(output_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(results, f, allow_unicode=True, sort_keys=False)
    print(f"Saved results to {output_yaml}")
    
    # Save to JSON file (if specified)
    if output_json:
        if os.path.dirname(output_json) and not os.path.exists(os.path.dirname(output_json)):
            os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Also saved results to {output_json}")

# Simulate human feedback
async def simulate_human_feedback(samples: List[Dict[str, Any]], 
                                representatives: List[Dict[str, Any]], 
                                correct_labels: Dict[Tuple[str, str], bool] = None,
                                human_accuracy: float = 1.0) -> List[Dict[str, Any]]:
    """
    Simulate human feedback on contradiction detection
    """
    import random
    corrected_pairs = []
    
    # Map from entity ID to index
    cluster_id_to_index = {rep["cluster_id"]: i for i, rep in enumerate(representatives)}
    
    for sample in samples:
        pair = sample["pair"]
        triplet = sample.get("triplet", None)
        
        # Get actual correct label (for simulation)
        actual_match = False
        if correct_labels and tuple(sorted(pair)) in correct_labels:
            actual_match = correct_labels[tuple(sorted(pair))]
        else:
            # If no correct labels provided, infer from data
            # E.g., if same original_cluster_id, then match
            try:
                idx1 = cluster_id_to_index.get(pair[0], -1)
                idx2 = cluster_id_to_index.get(pair[1], -1)
                if idx1 >= 0 and idx2 >= 0:
                    rep1 = representatives[idx1]
                    rep2 = representatives[idx2]
                    if rep1.get("original_cluster_id", "") == rep2.get("original_cluster_id", "") and rep1.get("original_cluster_id", ""):
                        actual_match = True
            except Exception as e:
                print(f"Error inferring correct label: {e}")
        
        # Simulate human answer based on accuracy
        human_answer = actual_match
        if random.random() > human_accuracy:
            # Error case: give opposite answer
            human_answer = not actual_match
        
        # Record correction
        correction = {
            "pair": pair,
            "triplet": triplet,
            "label": "match" if human_answer else "unmatch",
            "actual_match": actual_match,  # For debugging
            "correct_answer": human_answer == actual_match  # For debugging
        }
        
        corrected_pairs.append(correction)
    
    return corrected_pairs

# Update similarity results based on human feedback
def update_similarity_results(similarity_pairs: List[Dict[str, Any]], 
                             corrected_pairs: List[Dict[str, Any]],
                             prefer_merge: bool = True) -> None:
    """
    Update similarity results based on human feedback
    prefer_merge: Flag to prioritize merging (prevent fragmentation)
    """
    # Create pair mapping
    pair_to_index = {}
    for i, sim in enumerate(similarity_pairs):
        pair = tuple(sorted(sim["cluster_pair"]))
        pair_to_index[pair] = i
    
    # Apply corrections
    updated_count = 0
    for correction in corrected_pairs:
        pair = tuple(sorted(correction["pair"]))
        if pair in pair_to_index:
            idx = pair_to_index[pair]
            old_score = similarity_pairs[idx]["similarity_score"]
            
            # Update similarity
            if correction["label"] == "match":
                similarity_pairs[idx]["similarity_score"] = 1.0  # Perfect match
                similarity_pairs[idx]["reason"] = "Human feedback: match"
            else:
                # If preferring merges, use softer non-match
                if prefer_merge:
                    similarity_pairs[idx]["similarity_score"] = 0.3  # Not completely non-match
                    similarity_pairs[idx]["reason"] = "Human feedback: unmatch (soft)"
                else:
                    similarity_pairs[idx]["similarity_score"] = 0.0
                    similarity_pairs[idx]["reason"] = "Human feedback: unmatch"
            
            updated_count += 1
            print(f"Updated pair {pair} similarity from {old_score} to {similarity_pairs[idx]['similarity_score']}")

# Calculate year similarity
def calculate_year_similarity(date1: str, date2: str) -> float:
    """
    Calculate similarity between two date strings based on year
    
    Args:
        date1: First date string
        date2: Second date string
        
    Returns:
        Similarity score (0-1)
    """
    # Extract years
    year1 = extract_year(date1)
    year2 = extract_year(date2)
    
    # If both years cannot be extracted
    if not year1 or not year2:
        return 0.5  # Middle value if either is missing
    
    # Calculate year difference
    try:
        y1 = int(year1)
        y2 = int(year2)
        diff = abs(y1 - y2)
        
        # High similarity for small differences, low for large ones
        if diff == 0:
            return 1.0
        elif diff <= 1:
            return 0.9
        elif diff <= 2:
            return 0.8
        elif diff <= 5:
            return 0.7
        elif diff <= 10:
            return 0.5
        else:
            return 0.3
    except:
        return 0.5  # Middle value on conversion error

# Prepare training data for LLM
def prepare_training_data(feedback_data: List[Dict[str, Any]], 
                        representatives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create training dataset from human feedback
    
    Args:
        feedback_data: Accumulated human feedback
        representatives: Representative records list
        
    Returns:
        Training dataset
    """
    training_examples = []
    
    for feedback in feedback_data:
        # Get record IDs from pair
        pair_ids = feedback["pair"]
        
        # Get record information
        records = []
        for record_id in pair_ids:
            for rep in representatives:
                if rep["cluster_id"] == record_id:
                    records.append(rep)
                    break
        
        if len(records) != 2:
            continue  # Skip if records not found
        
        # Extract feature data (if available)
        features = feedback.get("features", {})
        
        # Feedback label
        is_match = feedback["label"] == "match"
        
        # Create training data
        example = {
            "messages": [
                {"role": "system", "content": "You are an expert in analyzing bibliographic record similarity."},
                {"role": "user", "content": f"""
                Analyze the similarity between these two book records and determine if they are the same book.
                Return a similarity score from 0 to 1.
                
                Record 1:
                Title: {records[0]['title']}
                Author: {records[0]['author']}
                Publisher: {records[0]['publisher']}
                Publication year: {records[0]['pubdate']}
                
                Record 2:
                Title: {records[1]['title']}
                Author: {records[1]['author']}
                Publisher: {records[1]['publisher']}
                Publication year: {records[1]['pubdate']}
                
                Consider title similarity, author name matches, publisher consistency, and other bibliographic factors.
                """},
                {"role": "assistant", "content": f"""
                I've analyzed the similarity between these records.
                
                Analysis:
                Title similarity: {features.get('title_similarity', 'Unknown')}
                Author similarity: {features.get('author_similarity', 'Unknown')}
                Publisher similarity: {features.get('publisher_similarity', 'Unknown')}
                Publication year similarity: {features.get('year_similarity', 'Unknown')}
                
                Conclusion: These records {'refer to the same book' if is_match else 'refer to different books'}.
                Similarity score: {1.0 if is_match else 0.0}
                
                Rationale:
                {get_similarity_reasoning(records[0], records[1], is_match)}
                """}
            ]
        }
        
        training_examples.append(example)
    
    return training_examples

# Generate similarity reasoning
def get_similarity_reasoning(record1: Dict[str, Any], record2: Dict[str, Any], is_match: bool) -> str:
    """
    Generate reasoning for similarity judgment between two records
    
    Args:
        record1: First record
        record2: Second record
        is_match: Whether they match
        
    Returns:
        Reasoning text
    """
    if is_match:
        # Reasoning for matching records
        reasons = []
        
        # Title similarity
        title_sim = calculate_text_similarity(record1['title'], record2['title'])
        if title_sim > 0.8:
            reasons.append(f"Titles are very similar (similarity: {title_sim:.2f})")
        elif title_sim > 0.6:
            reasons.append(f"Titles show moderate similarity (similarity: {title_sim:.2f})")
        
        # Author match
        author_sim = calculate_text_similarity(record1['author'], record2['author'])
        if author_sim > 0.8:
            reasons.append("Author names match")
        elif author_sim > 0.6:
            reasons.append("Author names have similar elements")
        
        # Publisher match
        publisher_sim = calculate_text_similarity(record1['publisher'], record2['publisher'])
        if publisher_sim > 0.8:
            reasons.append("Publishers match")
        
        # Year proximity
        year_sim = calculate_year_similarity(record1['pubdate'], record2['pubdate'])
        if year_sim > 0.8:
            reasons.append("Publication years match or are very close")
        
        # Default reason if none found
        if not reasons:
            reasons.append("The combination of overall features indicates the same book")
        
        return "・" + "\n・".join(reasons)
        
    else:
        # Reasoning for non-matching records
        reasons = []
        
        # Title differences
        title_sim = calculate_text_similarity(record1['title'], record2['title'])
        if title_sim < 0.4:
            reasons.append(f"Titles are significantly different (similarity: {title_sim:.2f})")
        
        # Author differences
        author_sim = calculate_text_similarity(record1['author'], record2['author'])
        if author_sim < 0.4:
            reasons.append("Author names are different")
        
        # Publisher differences
        publisher_sim = calculate_text_similarity(record1['publisher'], record2['publisher'])
        if publisher_sim < 0.4:
            reasons.append("Publishers are different")
        
        # Year differences
        year_sim = calculate_year_similarity(record1['pubdate'], record2['pubdate'])
        if year_sim < 0.5:
            reasons.append("Publication years have significant gap")
        
        # Default reason if none found
        if not reasons:
            reasons.append("There are distinguishing features of different books")
        
        return "・" + "\n・".join(reasons)

# Train LLM with feedback (simulated implementation)
async def train_llm_with_feedback(training_dataset_path: str, api_key: str) -> Dict[str, Any]:
    """
    Fine-tune LLM using human feedback
    
    Args:
        training_dataset_path: Path to training dataset file
        api_key: OpenAI API key
        
    Returns:
        Fine-tuning result
    """
    # Call fine-tuning API
    async with aiohttp.ClientSession() as session:
        # Upload training file
        upload_headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        # Upload file
        with open(training_dataset_path, 'rb') as f:
            form_data = aiohttp.FormData()
            form_data.add_field('purpose', 'fine-tune')
            form_data.add_field('file', f)
            
            async with session.post(
                "https://api.openai.com/v1/files", 
                data=form_data,
                headers=upload_headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"File upload error: {response.status} - {error_text}")
                
                file_result = await response.json()
                file_id = file_result["id"]
        
        # Create fine-tuning job
        tuning_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        tuning_payload = {
            "model": "gpt-4o-mini-2024-07-18",  # Base model (adjust as needed)
            "training_file": file_id,
            "suffix": f"book-matching-{time.strftime('%Y%m%d%H%M%S')}",  # Unique suffix
            "hyperparameters": {
                "n_epochs": 3
            }
        }
        
        async with session.post(
            "https://api.openai.com/v1/fine_tuning/jobs", 
            json=tuning_payload, 
            headers=tuning_headers
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Fine-tuning job creation error: {response.status} - {error_text}")
            
            result = await response.json()
            
            # Get job ID
            job_id = result["id"]
            print(f"Fine-tuning job started (ID: {job_id})")
            
            # Since API processes asynchronously, return simulated result
            simulated_result = {
                "id": job_id,
                "fine_tuned_model": f"ft:gpt-4o-mini-2024-07-18:book-matching:{int(time.time())}",
                "status": "succeeded",
                "created_at": int(time.time()),
                "training_file": file_id
            }
            
            return simulated_result

# Recalculate similarities with trained model
async def recalculate_similarities_with_trained_model(
    representatives: List[Dict[str, Any]],
    api_key: str,
    fine_tuned_model: str,
    output_dir: str = "results",
    output_prefix: str = "result",
    iteration: int = 0,
    strategy: str = "core_inconsistency",
    human_accuracy: float = 1.0
) -> List[Dict[str, Any]]:
    """
    ファインチューニングされたモデルを使用してレコード間の類似度を再計算
    個別処理方式（バッチサイズ=1）を使用
    """
    print(f"ファインチューニングされたモデル {fine_tuned_model} を使用して類似度を再計算中...")
    updated_similarities = []
    all_similarity_judgments = []
    
    # ペアを生成
    all_pairs = []
    for i, rep1 in enumerate(representatives):
        for j, rep2 in enumerate(representatives[i+1:], i+1):
            all_pairs.append((rep1, rep2))
    
    # 全ペア数を表示
    total_pairs = len(all_pairs)
    print(f"再計算対象ペア総数: {total_pairs}")
    
    # 処理ペア数を制限（必要に応じて）
    # max_pairs = min(1000, total_pairs)  # 最大1000ペアに制限する場合
    # all_pairs = all_pairs[:max_pairs]
    # print(f"処理を {max_pairs} ペアに制限します")
    
    # 処理カウンター
    processed = 0
    successful = 0
    error_count = 0
    
    # すべてのペアを個別に処理
    for rep1, rep2 in all_pairs:
        try:
            # プロンプトを準備
            prompt = f"""
            以下の2つの書籍レコードが同じ本を指しているかどうかを判断し、類似度スコア（0～1）を返してください：
            
            レコード1:
            タイトル: {rep1['title']}
            著者: {rep1['author']}
            出版社: {rep1.get('publisher', '不明')}
            出版年: {rep1.get('pubdate', '不明')}
            
            レコード2:
            タイトル: {rep2['title']}
            著者: {rep2['author']}
            出版社: {rep2.get('publisher', '不明')}
            出版年: {rep2.get('pubdate', '不明')}
            
            タイトルと著者の類似性、出版社と出版年情報を考慮して判断してください。
            最後に「類似度: X.X」の形式でスコアを返してください。
            """
            
            # API呼び出し
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": fine_tuned_model,
                    "messages": [
                        {"role": "system", "content": "あなたは書誌レコードの類似度を分析する専門家です。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0
                }
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                
                async with session.post(
                    "https://api.openai.com/v1/chat/completions", 
                    json=payload, 
                    headers=headers
                ) as response:
                    processed += 1
                    
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"API呼び出しエラー ({processed}/{total_pairs}): {response.status} - {error_text}")
                        error_count += 1
                        
                        # APIエラーの場合はデフォルト値を使用
                        similarity = {
                            "cluster_pair": [rep1["cluster_id"], rep2["cluster_id"]],
                            "similarity_score": 0.5,  # デフォルト値
                            "reason": f"APIエラー: {response.status}",
                            "records": [
                                {"cluster_id": rep1["cluster_id"], "title": rep1["title"], "author": rep1["author"]},
                                {"cluster_id": rep2["cluster_id"], "title": rep2["title"], "author": rep2["author"]}
                            ]
                        }
                        updated_similarities.append(similarity)
                        
                        # 判定結果も記録
                        all_similarity_judgments.append({
                            "record1": {
                                "cluster_id": rep1["cluster_id"],
                                "title": rep1["title"],
                                "author": rep1["author"],
                                "publisher": rep1.get("publisher", ""),
                                "pubdate": rep1.get("pubdate", "")
                            },
                            "record2": {
                                "cluster_id": rep2["cluster_id"],
                                "title": rep2["title"],
                                "author": rep2["author"],
                                "publisher": rep2.get("publisher", ""),
                                "pubdate": rep2.get("pubdate", "")
                            },
                            "similarity_score": 0.5,
                            "reason": f"APIエラー: {response.status}",
                            "model": fine_tuned_model
                        })
                        
                        # レート制限に遭遇した可能性がある場合は一時停止
                        if response.status in [429, 500, 503]:
                            print(f"レート制限またはサーバーエラーのため、一時停止します...")
                            await asyncio.sleep(5)  # 5秒待機
                        
                        continue
                    
                    result = await response.json()
                    successful += 1
                    
                    # 結果を処理
                    response_content = result["choices"][0]["message"]["content"]
                    
                    # スコアを抽出
                    score_match = re.search(r'類似度[:：]\s*(\d+\.\d+|\d+)', response_content)
                    if not score_match:
                        score_match = re.search(r'(\d+\.\d+|\d+)', response_content)
                    
                    if score_match:
                        score = float(score_match.group(1))
                        score = max(0.0, min(1.0, score))
                    else:
                        score = 0.5
                    
                    # 判定理由を抽出
                    reason = response_content.strip()
                    if len(reason) > 100:
                        reason = reason[:97] + "..."
                    
                    # 類似度エントリを作成
                    similarity = {
                        "cluster_pair": [rep1["cluster_id"], rep2["cluster_id"]],
                        "similarity_score": score,
                        "reason": reason,
                        "records": [
                            {"cluster_id": rep1["cluster_id"], "title": rep1["title"], "author": rep1["author"]},
                            {"cluster_id": rep2["cluster_id"], "title": rep2["title"], "author": rep2["author"]}
                        ]
                    }
                    updated_similarities.append(similarity)
                    
                    # 判定結果も記録
                    all_similarity_judgments.append({
                        "record1": {
                            "cluster_id": rep1["cluster_id"],
                            "title": rep1["title"],
                            "author": rep1["author"],
                            "publisher": rep1.get("publisher", ""),
                            "pubdate": rep1.get("pubdate", "")
                        },
                        "record2": {
                            "cluster_id": rep2["cluster_id"],
                            "title": rep2["title"],
                            "author": rep2["author"],
                            "publisher": rep2.get("publisher", ""),
                            "pubdate": rep2.get("pubdate", "")
                        },
                        "similarity_score": score,
                        "reason": reason,
                        "full_response": response_content,
                        "model": fine_tuned_model
                    })
        except Exception as e:
            print(f"ペア処理中にエラー ({processed}/{total_pairs}): {e}")
            error_count += 1
            
            # 例外が発生した場合はデフォルト値を使用
            similarity = {
                "cluster_pair": [rep1["cluster_id"], rep2["cluster_id"]],
                "similarity_score": 0.5,
                "reason": f"処理エラー: {str(e)}",
                "records": [
                    {"cluster_id": rep1["cluster_id"], "title": rep1["title"], "author": rep1["author"]},
                    {"cluster_id": rep2["cluster_id"], "title": rep2["title"], "author": rep2["author"]}
                ]
            }
            updated_similarities.append(similarity)
            
            # エラー時も判定結果を記録
            all_similarity_judgments.append({
                "record1": {
                    "cluster_id": rep1["cluster_id"],
                    "title": rep1["title"],
                    "author": rep1["author"],
                    "publisher": rep1.get("publisher", ""),
                    "pubdate": rep1.get("pubdate", "")
                },
                "record2": {
                    "cluster_id": rep2["cluster_id"],
                    "title": rep2["title"],
                    "author": rep2["author"],
                    "publisher": rep2.get("publisher", ""),
                    "pubdate": rep2.get("pubdate", "")
                },
                "similarity_score": 0.5,
                "reason": f"処理エラー: {str(e)}",
                "model": fine_tuned_model
            })
        
        # 進捗を表示
        if processed % 10 == 0 or processed == total_pairs:
            progress = processed / total_pairs * 100
            print(f"類似度再計算: {processed}/{total_pairs} ペア完了 ({progress:.1f}%) - 成功: {successful}, エラー: {error_count}")
        
        # レート制限対策で小さな遅延を入れる
        await asyncio.sleep(0.1)
        
        # 部分結果の保存（100ペアごと）
        if processed % 100 == 0:
            try:
                # 出力ディレクトリが存在しない場合は作成
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                # 部分結果を保存
                partial_file = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_iter{iteration}_partial_{processed}.json"
                with open(partial_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "processed_pairs": processed,
                        "total_pairs": total_pairs,
                        "successful": successful,
                        "error_count": error_count,
                        "similarities": updated_similarities,
                        "judgments": all_similarity_judgments
                    }, f, ensure_ascii=False, indent=2)
                print(f"部分結果を {partial_file} に保存しました")
            except Exception as e:
                print(f"部分結果保存中にエラー: {e}")
    
    # 結果をソート（類似度の高い順）
    updated_similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # すべての判定結果を保存
    all_judgments_file = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_iter{iteration}_all_judgments.json"
    print(f"再計算類似度判定を保存しています。出力先: {all_judgments_file}")
    print(f"判定結果数: {len(all_similarity_judgments)}")
    
    try:
        # 出力ディレクトリが存在しない場合は作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        with open(all_judgments_file, 'w', encoding='utf-8') as f:
            json.dump(all_similarity_judgments, f, ensure_ascii=False, indent=2)
        print(f"すべての類似度判定結果を {all_judgments_file} に保存しました（{len(all_similarity_judgments)}ペア）")
        
        # 判定エラーを分析して表示・保存
        try:
            analyze_judgment_errors(
                all_similarity_judgments, 
                representatives, 
                output_dir, 
                output_prefix, 
                strategy, 
                human_accuracy, 
                iteration
            )
        except Exception as e:
            print(f"判定エラー分析中にエラー: {e}")
        
    except Exception as e:
        print(f"判定結果保存中にエラーが発生: {e}")
    
    # 結果の統計情報を表示
    print(f"\n=== 類似度再計算 完了 ===")
    print(f"総ペア数: {total_pairs}")
    print(f"処理完了: {processed} ({processed/total_pairs*100:.1f}%)")
    print(f"成功: {successful} ({successful/total_pairs*100:.1f}%)")
    print(f"エラー: {error_count} ({error_count/total_pairs*100:.1f}%)")
    
    return updated_similarities


# Human-in-the-loop process with LLM learning
async def human_in_the_loop_process_with_llm_learning(yaml_data: str, 
                                   api_key: str, 
                                   batch_size: int = 300, 
                                   threshold: float = 0.7, 
                                   human_accuracy: float = 1.0,
                                   max_iterations: int = 10,
                                   strategy: str = 'core_inconsistency',
                                   learning_frequency: int = 3,  # How often to learn
                                   log_iterations: bool = False,
                                   output_prefix: str = "result",
                                   output_dir: str = "results") -> Dict[str, Any]:
    """
    Human-in-the-loop entity matching process with LLM learning
    
    Args:
        yaml_data: Input YAML data or file path
        api_key: OpenAI API key
        batch_size: API request batch size
        threshold: Clustering similarity threshold
        human_accuracy: Human answer accuracy simulation value
        max_iterations: Maximum iterations
        strategy: Sampling strategy
        learning_frequency: How often to perform LLM learning
        log_iterations: Whether to log detailed iteration info
        output_prefix: Output filename prefix
        output_dir: Output directory
        
    Returns:
        Processing result dictionary
    """
    print(f"=== Starting Human-in-the-loop process with LLM learning (strategy: {strategy}, human accuracy: {human_accuracy}) ===")
    
    # Determine if file path or content
    if isinstance(yaml_data, str) and os.path.exists(yaml_data):
        # File path - read file
        with open(yaml_data, 'r', encoding='utf-8') as f:
            yaml_content = f.read()
    else:
        # Already content - use as is
        yaml_content = yaml_data
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory {output_dir}")
    
    # 1. Extract representative records
    representatives = extract_representatives(yaml_content)
    print(f"Extracted {len(representatives)} cluster representatives")
    
    # 2. Extract all records
    data = yaml.safe_load(yaml_content)
    all_records = extract_records(data)
    print(f"Processing {len(all_records)} total records")
    
    # 3. Precompute features and initial similarity calculation
    features = precompute_features(representatives)
    
    # Divide into batches
    feature_batches = [features[i:i+batch_size] for i in range(0, len(features), batch_size)]
    
    # Prepare API requests
    api_requests = []
    for batch in feature_batches:
        request_data = create_efficient_api_request(batch)
        api_requests.append(request_data)
    
    # Initial API call
    api_results = await process_requests_parallel(api_requests, api_key)
    similarity_pairs = extract_similarity_results(
        api_results, 
        representatives, 
        output_dir, 
        output_prefix, 
        strategy, 
        human_accuracy
    )
    
    # Extract correct labels (for simulation)
    correct_labels = {}
    for i, rep1 in enumerate(representatives):
        for j, rep2 in enumerate(representatives):
            if i < j:  # Avoid duplicates
                # Consider match if same original cluster ID
                is_match = (rep1.get("original_cluster_id", "") == rep2.get("original_cluster_id", "") 
                          and rep1.get("original_cluster_id", ""))
                correct_labels[tuple(sorted([rep1["cluster_id"], rep2["cluster_id"]]))] = is_match
    
    # 4. Initial clustering
    clusters, _ = create_clusters_from_matches(similarity_pairs, representatives, threshold)
    
    # Variable to store accumulated feedback
    all_feedback = []
    
    # Track current model ID (initially default model)
    current_model = "gpt-4o-mini"
    
    # Track learning state
    learning_state = {
        "model_iterations": [],  # Iterations where new model used
        "feedback_counts": [],   # Feedback count per iteration
        "model_ids": []          # Model IDs used
    }
    
    # Track evaluation history
    evaluation_history = []
    
    # 5. Iteration process
    for iteration in range(max_iterations):
        print(f"\n----- Iteration {iteration+1}/{max_iterations} -----")
        
        # Evaluate current results
        output_groups = format_output_groups(clusters)
        metrics = calculate_output_metrics(output_groups, all_records)
        
        # Record iteration result
        iteration_result = {
            "iteration": iteration,
            "f1_pair": float(metrics.get("f1(pair)", "0")),
            "precision_pair": float(metrics.get("precision(pair)", "0")),
            "recall_pair": float(metrics.get("recall(pair)", "0")),
            "complete_group": float(metrics.get("complete(group)", "0")),
            "num_groups": metrics.get("num_of_groups(inference)", 0),
            "model": current_model
        }
        evaluation_history.append(iteration_result)
        
        print(f"Current F1 score: {metrics.get('f1(pair)', 'N/A')}")
        
        # Log iteration details (if enabled)
        if log_iterations:
            iteration_log_file = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_iter{iteration}_llm.json"
            with open(iteration_log_file, 'w', encoding='utf-8') as f:
                iteration_data = {
                    "iteration": iteration,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "metrics": metrics,
                    "model": current_model,
                    "clusters": {
                        "count": len(clusters),
                        "sizes": [len(cluster.get("all_records", [])) for cluster in clusters]
                    },
                    "feedback_count": len(all_feedback)
                }
                json.dump(iteration_data, f, ensure_ascii=False, indent=2)
            print(f"Saved iteration {iteration} information to {iteration_log_file}")
        
        # 5.1 Sample selection - find pairs for human feedback
        # This would typically involve detecting inconsistencies or uncertain pairs
        # Simplified implementation here
        samples = []
        # Logic to select samples based on strategy would go here
        # For example, finding pairs with similarity scores near the threshold
        for sim in similarity_pairs:
            score = sim["similarity_score"]
            if abs(score - threshold) < 0.2:  # Close to threshold
                samples.append({
                    "pair": sim["cluster_pair"],
                    "similarity_score": score
                })
        
        # Take only up to batch_size samples
        samples = samples[:batch_size]
        print(f"Selected {len(samples)} sample pairs for feedback")
        
        if not samples:
            print("No samples to evaluate. Ending process.")
            break
        
        # 5.2 Get human feedback (simulated)
        corrected_pairs = await simulate_human_feedback(
            samples, representatives, correct_labels, human_accuracy)
        
        print(f"Human feedback (accuracy: {human_accuracy}):")
        correct_answers = sum(1 for p in corrected_pairs if p["correct_answer"])
        print(f"- Total of {len(corrected_pairs)} pairs answered ({correct_answers} correct, accuracy: {correct_answers/len(corrected_pairs) if len(corrected_pairs) > 0 else 0:.2f})")
        
        # Format and accumulate feedback data
        for pair in corrected_pairs:
            # Add record information
            pair_records = []
            for record_id in pair["pair"]:
                for rep in representatives:
                    if rep["cluster_id"] == record_id:
                        pair_records.append(rep)
                        break
            
            # Only add if both records found
            if len(pair_records) == 2:
                # Enhance feedback with features
                enriched_feedback = {
                    "pair": pair["pair"],
                    "label": pair["label"],
                    "correct_answer": pair["correct_answer"],
                    "records": pair_records,
                    "features": {
                        "title_similarity": calculate_text_similarity(pair_records[0]["title"], pair_records[1]["title"]),
                        "author_similarity": calculate_text_similarity(pair_records[0]["author"], pair_records[1]["author"]),
                        "publisher_similarity": calculate_text_similarity(pair_records[0]["publisher"], pair_records[1]["publisher"]),
                        "year_similarity": calculate_year_similarity(pair_records[0]["pubdate"], pair_records[1]["pubdate"])
                    }
                }
                all_feedback.append(enriched_feedback)
        
        # Log correction info (if enabled)
        if log_iterations:
            corrections_log_file = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_iter{iteration}_corrections_llm.json"
            with open(corrections_log_file, 'w', encoding='utf-8') as f:
                json.dump(corrected_pairs, f, ensure_ascii=False, indent=2)
            print(f"Saved iteration {iteration} correction info to {corrections_log_file}")
        
        # 5.3 Update similarity results with feedback
        update_similarity_results(similarity_pairs, corrected_pairs)
        
        # 5.4 LLM learning (at specified frequency)
        if iteration > 0 and iteration % learning_frequency == 0 and len(all_feedback) >= 10:
            print(f"Learning from {len(all_feedback)} accumulated feedback instances...")
            
            try:
                # Save training dataset
                training_dataset_path = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_iter{iteration}_training_data.jsonl"
                training_examples = prepare_training_data(all_feedback, representatives)
                
                with open(training_dataset_path, 'w', encoding='utf-8') as f:
                    for example in training_examples:
                        f.write(json.dumps(example, ensure_ascii=False) + '\n')
                
                print(f"Saved training dataset ({len(training_examples)} examples) to {training_dataset_path}")
                
                # Fine-tune LLM
                fine_tuning_result = await train_llm_with_feedback(training_dataset_path, api_key)
                
                # Log fine-tuning result
                tuning_log_file = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_iter{iteration}_tuning_result.json"
                with open(tuning_log_file, 'w', encoding='utf-8') as f:
                    json.dump(fine_tuning_result, f, ensure_ascii=False, indent=2)
                
                # Get new model ID
                if "id" in fine_tuning_result and "fine_tuned_model" in fine_tuning_result:
                    new_model = fine_tuning_result["fine_tuned_model"]
                    print(f"Created new model: {new_model}")
                    
                    # Update learning state
                    learning_state["model_iterations"].append(iteration)
                    learning_state["feedback_counts"].append(len(all_feedback))
                    learning_state["model_ids"].append(new_model)
                    
                    # Update current model
                    current_model = new_model
                    
                    # Recalculate similarities with trained model
                    print("学習されたモデルで類似度を再計算中...")
                    updated_similarities = await recalculate_similarities_with_trained_model(
                        representatives, 
                        api_key, 
                        current_model,
                        output_dir,
                        output_prefix,
                        iteration,
                        strategy,
                        human_accuracy
                    )
                    
                    # Integrate old and new similarities
                    # Prioritize new judgments but keep high-confidence existing ones
                    updated_similarity_map = {
                        tuple(sorted(sim["cluster_pair"])): sim 
                        for sim in updated_similarities
                    }
                    
                    for i, sim in enumerate(similarity_pairs):
                        pair = tuple(sorted(sim["cluster_pair"]))
                        if pair in updated_similarity_map:
                            # Keep existing high-confidence judgments
                            existing_score = sim["similarity_score"]
                            if existing_score <= 0.1 or existing_score >= 0.9:
                                # High confidence judgment - keep it
                                updated_similarity_map[pair]["similarity_score"] = existing_score
                                updated_similarity_map[pair]["reason"] = f"{sim['reason']} (kept due to high confidence)"
                    
                    # Generate new similarity list from map
                    similarity_pairs = list(updated_similarity_map.values())
                    
                    print(f"Recalculated {len(similarity_pairs)} similarity pairs")
                    
                    # Log similarity changes
                    if log_iterations:
                        similarity_log_file = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_iter{iteration}_similarities.json"
                        with open(similarity_log_file, 'w', encoding='utf-8') as f:
                            json.dump(similarity_pairs, f, ensure_ascii=False, indent=2)
                        print(f"Saved recalculated similarities to {similarity_log_file}")
                    
            except Exception as e:
                print(f"Error during LLM learning: {e}")
                import traceback
                traceback.print_exc()
        
        # 5.5 Rebuild clusters
        clusters, merge_decisions = create_clusters_from_matches(similarity_pairs, representatives, threshold)

        # Log changes (if enabled)
        if log_iterations:
            changes_log_file = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_iter{iteration}_changes_llm.json"
            with open(changes_log_file, 'w', encoding='utf-8') as f:
                changes_data = {
                    "merge_decisions": merge_decisions,
                    "clusters": {
                        "count": len(clusters),
                        "sizes": [len(cluster.get("all_records", [])) for cluster in clusters]
                    },
                    "model": current_model
                }
                json.dump(changes_data, f, ensure_ascii=False, indent=2)
            print(f"Saved iteration {iteration} change info to {changes_log_file}")
    
    # Final evaluation
    output_groups = format_output_groups(clusters)
    final_metrics = calculate_output_metrics(output_groups, all_records)
    
    # Add final result to evaluation history
    evaluation_history.append({
        "iteration": max_iterations,
        "f1_pair": float(final_metrics.get("f1(pair)", "0")),
        "precision_pair": float(final_metrics.get("precision(pair)", "0")),
        "recall_pair": float(final_metrics.get("recall(pair)", "0")),
        "complete_group": float(final_metrics.get("complete(group)", "0")),
        "num_groups": final_metrics.get("num_of_groups(inference)", 0),
        "model": current_model
    })
    
    # Save iteration results to CSV
    if log_iterations:
        csv_file = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_iterations_llm.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            import csv
            writer = csv.writer(f)
            # Header row
            writer.writerow([
                "iteration", "f1_pair", "precision_pair", "recall_pair", 
                "complete_group", "num_groups", "model"
            ])
            
            # Results for each iteration
            for entry in evaluation_history:
                writer.writerow([
                    entry.get("iteration", 0),
                    entry.get("f1_pair", 0),
                    entry.get("precision_pair", 0),
                    entry.get("recall_pair", 0),
                    entry.get("complete_group", 0),
                    entry.get("num_groups", 0),
                    entry.get("model", "")
                ])
        print(f"Saved iteration summary to {csv_file}")
        
        # Save learning state
        learning_log_file = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_learning_state.json"
        with open(learning_log_file, 'w', encoding='utf-8') as f:
            json.dump(learning_state, f, ensure_ascii=False, indent=2)
        print(f"Saved learning state to {learning_log_file}")
    
    print("\n=== Human-in-the-loop process with LLM learning completed ===")
    print(f"Final F1 score: {final_metrics.get('f1(pair)', 'N/A')}")
    print(f"Model used: {current_model}")
    
    # Return results
    final_results = format_final_output(output_groups, final_metrics)
    final_results["evaluation_history"] = evaluation_history
    final_results["human_accuracy"] = human_accuracy
    final_results["strategy"] = strategy
    final_results["learning_state"] = learning_state
    final_results["model"] = current_model
    
    return final_results

def analyze_judgment_errors(
    all_judgments: List[Dict[str, Any]], 
    representatives: List[Dict[str, Any]], 
    output_dir: str = "results",
    output_prefix: str = "result",
    strategy: str = "core_inconsistency",
    human_accuracy: float = 1.0,
    iteration: int = 0
) -> None:
    """
    間違って判定されたペアを分析して表示・保存する
    
    Args:
        all_judgments: すべての判定結果
        representatives: 代表レコードのリスト
        output_dir: 出力ディレクトリ
        output_prefix: 出力プレフィックス 
        strategy: 戦略
        human_accuracy: 人間の精度
        iteration: 反復回数（初期判定の場合は-1）
    """
    print("\n=== 判定分析 ===")
    print(f"総ペア数: {len(all_judgments)}")
    
    # cluster_id から original_cluster_id へのマッピングを作成
    cluster_to_original = {}
    for rep in representatives:
        cluster_id = rep.get("cluster_id")
        original_id = rep.get("original_cluster_id", "")
        cluster_to_original[cluster_id] = original_id
        
    # デバッグ情報: マッピングのサンプルを表示
    print(f"代表レコード数: {len(representatives)}")
    print(f"cluster_id -> original_cluster_id マッピング例 (最大5件):")
    for i, (cluster_id, original_id) in enumerate(list(cluster_to_original.items())[:5]):
        print(f"  {cluster_id} -> {original_id}")
    
    # 各判定結果に対する分析
    analyzed_judgments = []
    original_id_based_matches = 0
    llm_matches = 0
    title_author_based_matches = 0
    
    for judgment in all_judgments:
        record1_id = judgment["record1"]["cluster_id"]
        record2_id = judgment["record2"]["cluster_id"]
        
        # LLMによる判定（閾値0.7）
        similarity_score = judgment["similarity_score"]
        llm_match = similarity_score >= 0.7
        if llm_match:
            llm_matches += 1
        
        # original_cluster_id による判定
        # cluster_id から original_cluster_id を取得
        original_id1 = cluster_to_original.get(record1_id, "unknown")
        original_id2 = cluster_to_original.get(record2_id, "unknown")
        
        original_id_match = (original_id1 == original_id2) and original_id1 and original_id1 != "unknown"
        if original_id_match:
            original_id_based_matches += 1
        
        # タイトルと著者ベースの判定（参考用）
        title1 = judgment["record1"]["title"]
        title2 = judgment["record2"]["title"]
        author1 = judgment["record1"]["author"]
        author2 = judgment["record2"]["author"]
        
        # 基本的な類似度計算
        title_similarity = calculate_text_similarity(title1, title2) if 'calculate_text_similarity' in globals() else 0.5
        author_similarity = calculate_text_similarity(author1, author2) if 'calculate_text_similarity' in globals() else 0.5
        
        # タイトルと著者の類似度が高い場合は同じ本と見なす
        title_author_match = title_similarity > 0.8 and author_similarity > 0.7
        if title_author_match:
            title_author_based_matches += 1
        
        # 分析結果を記録
        analyzed_judgment = {
            "judgment": judgment,
            "record1": {
                "cluster_id": record1_id,
                "original_cluster_id": original_id1,
                "title": title1,
                "author": author1
            },
            "record2": {
                "cluster_id": record2_id,
                "original_cluster_id": original_id2,
                "title": title2,
                "author": author2
            },
            "llm_match": llm_match,
            "original_id_match": original_id_match,
            "title_author_match": title_author_match,
            "similarity_score": similarity_score,
            "title_similarity": title_similarity,
            "author_similarity": author_similarity
        }
        
        analyzed_judgments.append(analyzed_judgment)
    
    # 分析結果を整理
    false_positives_orig = []  # original_id基準での偽陽性
    false_negatives_orig = []  # original_id基準での偽陰性
    false_positives_ta = []    # タイトル・著者基準での偽陽性
    false_negatives_ta = []    # タイトル・著者基準での偽陰性
    
    for judgment in analyzed_judgments:
        # original_id基準での誤判定
        if judgment["llm_match"] and not judgment["original_id_match"]:
            false_positives_orig.append(judgment)
        elif not judgment["llm_match"] and judgment["original_id_match"]:
            false_negatives_orig.append(judgment)
        
        # タイトル・著者基準での誤判定
        if judgment["llm_match"] and not judgment["title_author_match"]:
            false_positives_ta.append(judgment)
        elif not judgment["llm_match"] and judgment["title_author_match"]:
            false_negatives_ta.append(judgment)
    
    # 分析結果の概要を表示
    print(f"\n=== 判定分析結果 ===")
    print(f"総ペア数: {len(all_judgments)}")
    print(f"LLMが一致と判定: {llm_matches} ペア ({llm_matches/len(all_judgments)*100:.1f}%)")
    print(f"original_idが一致: {original_id_based_matches} ペア ({original_id_based_matches/len(all_judgments)*100:.1f}%)")
    print(f"タイトル・著者が類似: {title_author_based_matches} ペア ({title_author_based_matches/len(all_judgments)*100:.1f}%)")
    
    print(f"\n-- original_id基準の誤判定 --")
    print(f"偽陽性 (False Positives): {len(false_positives_orig)} ペア (original_idでは不一致だがLLMは一致と判定)")
    print(f"偽陰性 (False Negatives): {len(false_negatives_orig)} ペア (original_idでは一致だがLLMは不一致と判定)")
    
    print(f"\n-- タイトル・著者基準の誤判定 --")
    print(f"偽陽性 (False Positives): {len(false_positives_ta)} ペア (タイトル・著者は類似していないがLLMは一致と判定)")
    print(f"偽陰性 (False Negatives): {len(false_negatives_ta)} ペア (タイトル・著者は類似しているがLLMは不一致と判定)")
    
    # 矛盾するトリプルを検出
    try:
        inconsistent_triplets = detect_transitivity_violations(all_judgments, 0.7)
        print(f"\n矛盾するトリプル: {len(inconsistent_triplets)} 件")
    except Exception as e:
        print(f"矛盾トリプル検出中にエラーが発生: {e}")
        inconsistent_triplets = []
    
    # 詳細な例を表示
    if false_positives_orig:
        print("\n== original_id基準の偽陽性の例（最大5件） ==")
        for i, fp in enumerate(false_positives_orig[:5]):
            judgment = fp["judgment"]
            print(f"{i+1}. 「{fp['record1']['title']}」と「{fp['record2']['title']}」")
            print(f"   著者: {fp['record1']['author']} / {fp['record2']['author']}")
            print(f"   類似度: {fp['similarity_score']}")
            print(f"   original_cluster_id: {fp['record1']['original_cluster_id']} / {fp['record2']['original_cluster_id']}")
    
    if false_negatives_orig:
        print("\n== original_id基準の偽陰性の例（最大5件） ==")
        for i, fn in enumerate(false_negatives_orig[:5]):
            judgment = fn["judgment"]
            print(f"{i+1}. 「{fn['record1']['title']}」と「{fn['record2']['title']}」")
            print(f"   著者: {fn['record1']['author']} / {fn['record2']['author']}")
            print(f"   類似度: {fn['similarity_score']}")
            print(f"   original_cluster_id: {fn['record1']['original_cluster_id']} / {fn['record2']['original_cluster_id']}")
    
    # 結果をファイルに保存
    iteration_str = "initial" if iteration == -1 else f"iter{iteration}"
    analysis_file = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_{iteration_str}_analysis.json"
    
    analysis_results = {
        "total_judgments": len(all_judgments),
        "llm_matches": llm_matches,
        "original_id_matches": original_id_based_matches,
        "title_author_matches": title_author_based_matches,
        "original_id_false_positives": {
            "count": len(false_positives_orig),
            "examples": false_positives_orig[:20]
        },
        "original_id_false_negatives": {
            "count": len(false_negatives_orig),
            "examples": false_negatives_orig[:20]
        },
        "title_author_false_positives": {
            "count": len(false_positives_ta),
            "examples": false_positives_ta[:20]
        },
        "title_author_false_negatives": {
            "count": len(false_negatives_ta),
            "examples": false_negatives_ta[:20]
        },
        "inconsistent_triplets": {
            "count": len(inconsistent_triplets),
            "examples": inconsistent_triplets[:20]
        },
        "all_judgments": analyzed_judgments
    }
    
    try:
        # 出力ディレクトリが存在しない場合は作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        print(f"\n判定分析結果を {analysis_file} に保存しました")
        
        # マッピング情報だけを別ファイルにも保存（デバッグ用）
        mapping_file = f"{output_dir}/{output_prefix}_{strategy}_{int(human_accuracy*100)}_{iteration_str}_id_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            mapping_data = {
                "cluster_to_original": cluster_to_original,
                "representative_count": len(representatives),
                "mapped_count": len(cluster_to_original)
            }
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        print(f"ID対応マッピングを {mapping_file} に保存しました")
        
    except Exception as e:
        print(f"分析結果保存中にエラーが発生: {e}")

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    2つのテキスト間の類似度を計算（簡易版）
    
    Args:
        text1: 比較する1つ目のテキスト
        text2: 比較する2つ目のテキスト
        
    Returns:
        類似度スコア（0〜1）
    """
    if not text1 or not text2:
        return 0.0
    
    # jellyfish.jaro_winkler_similarityを使用していない場合の代替処理
    try:
        from jellyfish import jaro_winkler_similarity
        return jaro_winkler_similarity(text1, text2)
    except ImportError:
        # 簡易的な類似度計算（文字の共通性に基づく）
        chars1 = set(text1)
        chars2 = set(text2)
        common = len(chars1.intersection(chars2))
        total = len(chars1.union(chars2))
        return common / total if total > 0 else 0.0
    
    
    
def detect_transitivity_violations(judgments: List[Dict[str, Any]], threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    判定結果から推移律に違反するトリプルを検出
    
    Args:
        judgments: 判定結果リスト
        threshold: 一致と見なす閾値
        
    Returns:
        矛盾するトリプルのリスト
    """
    # グラフ構造を構築
    match_graph = defaultdict(set)
    unmatch_graph = defaultdict(set)
    
    # ペアごとに一致/不一致グラフに追加
    for judgment in judgments:
        record1_id = judgment["record1"]["cluster_id"]
        record2_id = judgment["record2"]["cluster_id"]
        similarity_score = judgment["similarity_score"]
        
        if similarity_score >= threshold:
            match_graph[record1_id].add(record2_id)
            match_graph[record2_id].add(record1_id)
        else:
            unmatch_graph[record1_id].add(record2_id)
            unmatch_graph[record2_id].add(record1_id)
    
    # 推移律に違反するトリプルを検出
    inconsistent_triplets = []
    
    # すべてのノードの組み合わせを調べる
    nodes = list(match_graph.keys())
    for i, node_a in enumerate(nodes):
        for node_b in match_graph[node_a]:
            if node_b <= node_a:  # 重複チェックを避ける
                continue
                
            for node_c in match_graph[node_b]:
                if node_c <= node_b:  # 重複チェックを避ける
                    continue
                
                # A=B, B=C, A≠C という矛盾パターンを検出
                if node_c in unmatch_graph.get(node_a, set()) or node_a in unmatch_graph.get(node_c, set()):
                    # 矛盾するトリプル情報を構築
                    triplet_info = {
                        "nodes": [node_a, node_b, node_c],
                        "relationships": {
                            f"{node_a}-{node_b}": "match",
                            f"{node_b}-{node_c}": "match",
                            f"{node_a}-{node_c}": "unmatch"
                        },
                        "titles": {},
                        "authors": {}
                    }
                    
                    # タイトルと著者情報を追加
                    for judgment in judgments:
                        rec1_id = judgment["record1"]["cluster_id"]
                        rec2_id = judgment["record2"]["cluster_id"]
                        
                        if rec1_id in triplet_info["nodes"] and rec1_id not in triplet_info["titles"]:
                            triplet_info["titles"][rec1_id] = judgment["record1"]["title"]
                            triplet_info["authors"][rec1_id] = judgment["record1"]["author"]
                        
                        if rec2_id in triplet_info["nodes"] and rec2_id not in triplet_info["titles"]:
                            triplet_info["titles"][rec2_id] = judgment["record2"]["title"]
                            triplet_info["authors"][rec2_id] = judgment["record2"]["author"]
                    
                    inconsistent_triplets.append(triplet_info)
    
    return inconsistent_triplets




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='改善された複数代表レコードによるエンティティマッチング')
    
    # 基本的な引数
    parser.add_argument('--input', '-i', type=str, required=True, help='処理するYAMLファイルのパス')
    parser.add_argument('--output', '-o', type=str, help='結果を保存するYAMLファイルのパス')
    parser.add_argument('--api-key', '-k', type=str, help='OpenAI APIキー（環境変数が設定されていない場合）')
    parser.add_argument('--threshold', '-t', type=float, default=0.7, help='類似度しきい値')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='APIリクエストのバッチサイズ')
    parser.add_argument('--similarity-report', '-s', type=str, help='類似度レポートを別途保存するJSONファイルのパス')
    
    # Human-in-the-loop関連の引数
    parser.add_argument('--strategy', '-st', type=str, default='core_inconsistency', 
                     choices=['core_inconsistency', 'inconsistency', 'uncertainty', 'hybrid'],
                     help='サンプリング戦略の選択')
    parser.add_argument('--human-accuracy', '-ha', type=float, default=1.0,
                     help='人間の回答精度のシミュレーション（0.0-1.0）')
    parser.add_argument('--iterations', '-it', type=int, default=10, 
                     help='Human-in-the-loopの最大反復回数')
    
    # LLM学習関連の引数（新規追加）
    parser.add_argument('--llm-learning', '-ll', action='store_true',
                     help='LLM学習機能を有効化')
    parser.add_argument('--learning-frequency', '-lf', type=int, default=3,
                     help='LLM学習を行う反復頻度（デフォルト: 3）')
    
    # 複数代表モード用の引数
    parser.add_argument('--multi-rep', '-mr', action='store_true', 
                     help='より正確なマッチングのためにクラスターごとに複数の代表レコードを使用')
    parser.add_argument('--reps-per-cluster', '-rpc', type=int, default=5,
                     help='クラスターごとの最大代表レコード数（デフォルト: 5）')
    
    # 出力設定関連の引数
    parser.add_argument('--output-dir', '-od', type=str, default='results',
                     help='結果ファイルを保存するディレクトリ（デフォルト: results）')
    parser.add_argument('--log-iterations', '-li', action='store_true', 
                     help='各反復の詳細をログファイルに出力')
    parser.add_argument('--generate-graphs', '-gg', action='store_true',
                     help='反復結果のグラフを自動生成')
    
    # デバッグ関連の引数
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグモードを有効化')
    
    args = parser.parse_args()
    
    # デバッグプリント追加: コマンドライン引数で指定された閾値
    print(f"\n[DEBUG] コマンドライン引数で指定された閾値: {args.threshold}")
    
    #suppress_font_warnings()
    # 出力ディレクトリの作成
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"出力ディレクトリ {args.output_dir} を作成しました")
    
    # APIキーの取得
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("警告: OpenAI APIキーが指定されていません。")
        exit(1)
    
    # 入力ファイル名からプレフィックスを作成
    input_basename = os.path.basename(args.input)
    input_prefix = os.path.splitext(input_basename)[0]
    
    # デバッグプリント追加: 処理実行前の閾値の確認
    print(f"[DEBUG] 処理実行前の閾値: {args.threshold}")
    
    # ファイル処理の実行
    if args.llm_learning:
        # LLM学習ありのHuman-in-the-loopプロセスを実行
        print(f"[DEBUG] human_in_the_loop_process_with_llm_learning 実行開始 (閾値: {args.threshold})")
        result = asyncio.run(human_in_the_loop_process_with_llm_learning(
            yaml_data=args.input,  # ファイルパスをそのまま渡す
            api_key=api_key,
            batch_size=args.batch_size,
            threshold=args.threshold,  # この閾値の引き継ぎを確認
            human_accuracy=args.human_accuracy,
            max_iterations=args.iterations,
            strategy=args.strategy,
            learning_frequency=args.learning_frequency,
            log_iterations=args.log_iterations,
            output_prefix=input_prefix,
            output_dir=args.output_dir
        ))
        print(f"[DEBUG] human_in_the_loop_process_with_llm_learning 実行完了")
        
        # 結果を保存
        output_file = args.output or f"{args.output_dir}/{input_prefix}_{args.strategy}_{int(args.human_accuracy*100)}_llm.yaml"
        output_json = f"{os.path.splitext(output_file)[0]}.json"
        save_result_files(result, output_file, output_json)
        
        """
        # グラフを生成
        if args.generate_graphs:
            # LLM学習用に拡張したグラフ生成関数
            generate_llm_learning_graphs(
                input_prefix,
                args.strategy,
                args.human_accuracy,
                args.output_dir
            )
            """
    else:
        # 従来のプロセスを実行
        print(f"[DEBUG] process_yaml_file 実行開始 (閾値: {args.threshold})")
        result = asyncio.run(process_yaml_file(
            args.input, 
            args.output, 
            args.batch_size, 
            args.threshold,  # この閾値の引き継ぎを確認
            api_key,
            strategy=args.strategy,
            human_accuracy=args.human_accuracy,
            max_iterations=args.iterations,
            similarity_report_path=args.similarity_report,
            multi_rep=args.multi_rep,
            reps_per_cluster=args.reps_per_cluster,
            log_iterations=args.log_iterations,
            output_dir=args.output_dir
        ))
        print(f"[DEBUG] process_yaml_file 実行完了")
    """
        # グラフを生成
        if args.generate_graphs and "evaluation_history" in result:
            generate_iteration_graphs(
                input_prefix,
                args.strategy,
                args.human_accuracy,
                args.output_dir
            )
    """
    print("処理が完了しました。")