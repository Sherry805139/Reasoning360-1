import re
from .data import Data
from .verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END


class ObjectCountingVerifier(Verifier):
    """
    验证器用于物品计数游戏的答案是否正确
    """
    def verify(self, data: Data, test_answer: str):
        try:
            ground_truth = int(data.answer)
            parsed_answer = self.extract_answer(test_answer)
            with open("solution_str_OC.txt", "a") as f:
                    f.write("data.answer: " + data.answer + '\n')
                    f.write("test_answer: " + test_answer + '\n')
                    f.write("parsed_answer" + parsed_answer + '\n')
                    f.write('-'*32 + '\n')
            
            if parsed_answer is None:
                return False
            return int(parsed_answer) == ground_truth

        except Exception as e:
            return False
    
    def extract_answer(self, answer_str):
        # 先找到最后一个\boxed{的位置
        last_box_index = answer_str.rfind("\\boxed{")
        
        if last_box_index == -1:
            return None
            
        # 从最后一个\Box{开始截取字符串
        last_box_substring = answer_str[last_box_index:]
        
        # 在截取的子字符串中进行正则匹配
        box_pattern = r'\\boxed\{([^}]*)\}'
        match = re.search(box_pattern, last_box_substring)
        
        if match:
            return match.group(1).strip()
        return answer_str
        