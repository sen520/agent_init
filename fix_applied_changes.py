#!/usr/bin/env python3
"""修复applied_changes问题的脚本"""

import os
import sys

def fix_state_base():
    """修复src/state/base.py文件"""
    file_path = "src/state/base.py"
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到 State 类定义后的合适位置
    lines = content.split('\n')
    
    # 找到 current_implementation 行
    for i, line in enumerate(lines):
        if 'current_implementation:' in line:
            # insert after 3 lines (current_implementation + implementations lines)
            insert_position = i + 3
            break
    else:
        print("❌ 找不到 current_implementation 字段")
        return False
    
    # 插入 applied_changes 字段
    indent = ' ' * 8  # 根据现有源码的缩进
    new_line = f'{indent}applied_changes: List[str] = Field(default_factory=list)  # 应用的具体变更'
    
    lines.insert(insert_position, new_line)
    
    # 验证修复
    new_content = '\n'.join(lines)
    if 'applied_changes: List[str]' not in new_content:
        print("❌ 修复失败")
        return False
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ 修复成功: {file_path}")
    return True

def test_fix():
    """测试修复"""
    print("🧪 测试修复...")
    try:
        # 直接测试State类
        with open("src/state/base.py", 'r', encoding='utf-8') as f:
            if 'applied_changes: List[str]' in f.read():
                print("✅ applied_changes字段已添加")
                return True
            else:
                print("❌ applied_changes字段未找到")
                return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    print("🔧 修复applied_changes问题")
    print("=" * 60)
    
    if fix_state_base():
        if test_fix():
            print("\n✅ 修复完成!")
            return True
        else:
            print("\n⚠️ 修复似乎不完整")
            return False
    else:
        print("\n❌ 修复失败")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)