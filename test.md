### Ubuntu

使用 apt-get 安装 xclip

```
sudo apt-get install xclip -y
```

注册 alias 到 `~/.bash_profile`

```
vim ~/.bashrc
# 添加内容
alias pbcopy='xclip -selection clipboard'
alias pbpaste='xclip -selection clipboard -o'
```

载入 `~/.bash_profile`

```
source ~/.bashrc
```

使用 pbcopy 将文件内容复制到剪切板

```
pbcopy < key_words.txt
```

### Centos

使用 yum 安装 xclip

```
sudo yum install xclip
```

注册 alias 到 `~/.bash_profile`

```
vim ~/.bash_profile
# 添加内容
alias pbcopy='xclip -selection clipboard'
alias pbpaste='xclip -selection clipboard -o'
```

载入 `~/.bash_profile`

```
source ~/.bash_profile
```

使用 pbcopy 将文件内容复制到剪切板

```
pbcopy < key_words.txt
```

