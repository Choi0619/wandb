import json

# corpus.json 파일을 읽어서 처리하는 함수
def count_data_pairs(file_path='corpus.json'):
    try:
        # JSON 파일 열기
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 데이터 페어 카운트
        count = 0
        for entry in data:
            if "role" in entry and "content" in entry:
                count += 1
        
        print(f"Total number of data pairs: {count}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# 함수 실행
if __name__ == "__main__":
    count_data_pairs()
