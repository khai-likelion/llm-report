import json
import os
from openai import OpenAI
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
import prompts

# .env 파일 로드
load_dotenv()

class GPTReportGenerator:
    def __init__(self, split_data_dir, pop_path):
        # API Key를 환경 변수에서 로드
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY가 환경 변수에 설정되지 않았습니다. .env 파일을 확인하세요.")
        
        self.client = OpenAI(api_key=api_key)
        self.split_data_dir = split_data_dir
        self.pop_path = pop_path
        self.output_dir = "generated_reports"
        self.pop_data = self._load_population_data()
        self.store_file_map = self._get_store_file_map()
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _load_population_data(self):
        """인구 데이터 로드"""
        try:
            with open(self.pop_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 인구 데이터 로드 중 오류 발생: {e}")
            return {}

    def _get_store_file_map(self):
        """디렉토리 내의 파일명과 매장명 매핑 (공백 제거 기준)"""
        mapping = {}
        try:
            if not os.path.exists(self.split_data_dir):
                print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {self.split_data_dir}")
                return {}
            
            for filename in os.listdir(self.split_data_dir):
                if filename.endswith(".json"):
                    store_name = filename.replace(".json", "").replace(" ", "")
                    mapping[store_name] = filename
            
            print(f"✅ 파일 시스템 스캔 완료: 총 {len(mapping)}개 매장 파일 식별")
            return mapping
        except Exception as e:
            print(f"❌ 파일 스캔 중 오류 발생: {e}")
            return {}

    def _load_store_data(self, store_name):
        """특정 매장의 데이터를 로드하고 인구 데이터 및 필드 매핑 결합"""
        search_key = store_name.replace(" ", "")
        filename = self.store_file_map.get(search_key)
        
        if not filename:
            return None

        try:
            file_path = os.path.join(self.split_data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 필드 매핑 및 통합 (기존 프롬프트 호환성 유지)
            area = data.get('metadata', {}).get('area', '망원동')
            
            # 인구 데이터 결합
            pop_info = "데이터 부족"
            if isinstance(self.pop_data, dict):
                pop_info = self.pop_data.get(area, self.pop_data)
            elif isinstance(self.pop_data, list):
                pop_info = next((p for p in self.pop_data if p.get('area') == area), 
                                self.pop_data[0] if self.pop_data else "데이터 부족")

            # 리포트 엔진에서 기대하는 필드명으로 매핑
            data.update({
                'population_data': pop_info,
                'sales_metrics': data.get('revenue_analysis', "데이터 부족"),
                'industry_context': data.get('market_analysis', {})
            })
            
            return data
        except Exception as e:
            print(f"❌ '{store_name}' 데이터 로드 중 오류 발생: {e}")
            return None

    def generate_report(self, store_name):
        data = self._load_store_data(store_name)
        search_key = store_name.replace(" ", "")

        if not data:
            return f"❌ '{store_name}' 매장 데이터를 찾을 수 없습니다."

        # --- [Step 1: 데이터 분석 및 추론 (Temperature 0.1)] ---
        # [변경] 지능적인 분석을 위해 시스템 프롬프트에 데이터 사이언티스트 페르소나 부여
        print(f"🧠 Step 1: GPT-5.2 심층 분석 중...")
        
        # 프롬프트 구성
        system_prompt = prompts.ANALYSIS_SYSTEM_PROMPT
        user_prompt = prompts.ANALYSIS_USER_TEMPLATE.format(
            store_name=data.get('store_name'),
            review_metrics=json.dumps(data.get('review_metrics'), ensure_ascii=False),
            population_data=data.get('population_data', '데이터 부족'),
            sales_metrics=data.get('sales_metrics', '데이터 부족'),
            industry_context=json.dumps(data.get('industry_context', '데이터 부족'), ensure_ascii=False),
            critical_feedback=data.get('critical_feedback')
        )

        analysis_response = self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-5.2"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1 # [변경] 분석의 정확성을 위해 낮은 온도 설정
        )
        analysis_result = analysis_response.choices[0].message.content

        # --- [Step 2: 리포트 시각화 및 생성 (X-report 전용 프롬프트)] ---
        print(f"✍️ Step 2: X-report 생성 및 솔루션 도출 중...")
        
        # 프롬프트 구성
        report_system_prompt = prompts.REPORT_SYSTEM_PROMPT
        report_user_prompt = prompts.REPORT_USER_TEMPLATE.format(
            store_name=store_name,
            analysis_result=analysis_result,
            avg_price=data.get('객단가', '정보 부족')
        )

        report_response = self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-5.2"),
            messages=[
                {"role": "system", "content": report_system_prompt},
                {"role": "user", "content": report_user_prompt}
            ],
            temperature=0.7 
        )
        report_text = report_response.choices[0].message.content

        # --- [파일 저장 로직] ---
        # [변경] JSON과 MD 파일을 동시에 저장하여 관리 및 문서화 용이성 확보
        result_json = {
            "store_name": store_name,
            "input_data": data,
            "analysis_log": analysis_result, 
            "output_report": report_text
        }
        
        json_path = os.path.join(self.output_dir, f"{search_key}_result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=4)
        
        md_path = os.path.join(self.output_dir, f"{search_key}_report.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n💾 파일 저장 완료: {json_path}, {md_path}")
        
        # [추가] 시뮬레이션을 위한 구조화된 솔루션 추출
        suggested_solutions = self.extract_solutions(report_text)
        return report_text, suggested_solutions

    def extract_solutions(self, report_text: str) -> List[Dict]:
        """리포트 텍스트에서 시뮬레이션용 수치 솔루션 추출 (새로운 카테고리형 양식 대응)"""
        solutions = []
        # 메트릭 매핑 (테마 및 키워드 기반)
        mapping = {
            "공간 경험": "insta",
            "제품 매력": "taste",
            "접점 확대": "overall",
            "재방문 락인": "service",
            "브랜드 인지": "insta",
            "맛": "taste",
            "가격": "price_value",
            "가성비": "price_value",
            "서비스": "service",
            "청결": "service",
            "분위기": "insta",
            "인테리어": "insta"
        }
        
        import re
        # 카테고리별 섹션 분리 (GPT 출력이 **[테마]** 형태를 쓸 수 있으므로 유연하게 매칭)
        categories = re.split(r"### 🔹 전략 카테고리 \d+[:.]\s*\*{0,2}\[?(.*?)\]?\*{0,2}\s*(?:\(.*?\))?\s*\n", report_text)
        
        # categories[0]은 앞부분, 이후 [테마, 내용, 테마, 내용, ...] 순서
        for i in range(1, len(categories), 2):
            theme = categories[i].strip().strip('*').strip('[').strip(']').strip()
            content = categories[i+1] if i+1 < len(categories) else ""
            
            # 기본 메트릭 결정
            base_metric = "overall"
            for keyword, target in mapping.items():
                if keyword in theme:
                    base_metric = target
                    break
            
            # 해당 카테고리 내 개별 솔루션 추출 (**솔루션 X: 또는 - 솔루션 X: 형태 모두 지원)
            sols = re.findall(r"-\s*\*{0,2}솔루션 [A-Z][:.]\*{0,2}\s*(.*?)(?=\n-\s*\*{0,2}솔루션 [A-Z]|### |## |$)", content, re.DOTALL)
            for sol_text in sols:
                sol_name = sol_text.strip().split('\n')[0].strip().strip('*')
                if not sol_name or len(sol_name) < 3:
                    continue
                
                # 솔루션 텍스트 내 키워드로 메트릭 재정의 (필요시)
                sol_metric = base_metric
                for keyword, target in mapping.items():
                    if keyword in sol_name:
                        sol_metric = target
                        break
                
                solutions.append({
                    "name": sol_name,
                    "metric": sol_metric,
                    "impact": 0.15 # 기본 임팩트
                })
        
        # 폴백: 정규식에 매칭되지 않는 경우, 더 넓은 패턴으로 재시도
        if not solutions:
            fallback = re.findall(r"솔루션 [A-Z][:.]\s*\*{0,2}(.*?)\*{0,2}\s*\n", report_text)
            for sol_name in fallback:
                sol_name = sol_name.strip().strip('*')
                if not sol_name or len(sol_name) < 3:
                    continue
                sol_metric = "overall"
                for keyword, target in mapping.items():
                    if keyword in sol_name:
                        sol_metric = target
                        break
                solutions.append({
                    "name": sol_name,
                    "metric": sol_metric,
                    "impact": 0.15
                })
            
        return solutions

# --- 실행부 ---
if __name__ == "__main__":
    split_dir = r'c:\Users\changhyun\Desktop\New_KHAI\agent-sim\data\raw\split_by_store_id_ver3'
    pop_db = r'c:\Users\changhyun\Desktop\New_KHAI\agent-sim\data\raw\인구_DB.json'
    
    # API Key는 내부에서 환경 변수로 로드하므로 인자에서 제거 가능하나,
    # 기존 __init__ 호환성을 위해 수정 필요.
    # __init__에서 api_key 인자를 제거하고 내부에서 로드하도록 변경했으므로 호출부도 수정.
    
    try:
        generator = GPTReportGenerator(split_dir, pop_db)
        target_store = input("🔎 리포트를 생성할 매장명을 입력하세요: ")

        print(f"\n🚀 GPT-5.2가 '{target_store}' 분석 및 생성을 시작합니다...\n")
        report_text, solutions = generator.generate_report(target_store)
        print("-" * 30 + "\n" + report_text + "\n" + "-" * 30)
    except Exception as e:
        print(f"⚠️ 오류 발생: {e}")

