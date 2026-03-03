# 开发安全规范

## 提交前检查清单

在每次 `git commit` 之前，必须检查以下内容：

### ✅ 敏感信息检查
- [ ] **API 密钥**: OpenAI API Key、数据库密码、SSH 私钥等
- [ ] **配置文件**: `.env`、`.env.local`、配置文件中的硬编码密钥
- [ ] **个人数据**: 用户名、邮箱、IP 地址、内部 URL
- [ ] **临时文件**: 日志文件、缓存文件、备份文件

### ✅ 文件类型检查
```bash
# 检查即将提交的文件
git diff --cached --name-only

# 检查是否有敏感内容
git diff --cached | grep -i "api_key\|password\|secret\|token\|private"
```

### ✅ 常见敏感文件（应已在 .gitignore 中）
```
.env
.env.local
.env.*.local
*.pem
*.key
secrets/
credentials/
*.backup
.optimization_backups/
```

### ✅ 提交前命令
```bash
# 1. 查看要提交的文件
git status

# 2. 检查 diff 内容
git diff --cached

# 3. 确认无误后提交
git commit -m "[code by kimiclaw] 你的提交信息"
```

---

## 历史记录

**2026-03-04**: 用户提醒提交前检查敏感信息，已修复 .gitignore 排除备份文件。
