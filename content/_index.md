---
title: 我的笔记
toc: false
---

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Blog</title>
    <link rel="stylesheet" href="{{ .Site.BaseURL }}css/style.css">
</head>
<body>
    <header>
        <h1>{{ .Site.Title }}</h1>
    </header>
    <main>
        <h2>Blog Posts</h2>
        <ul>
            {{ range .Site.RegularPages }}
                <li>
                    <a href="{{ .Permalink }}">{{ .Title }}</a>
                </li>
            {{ end }}
        </ul>
    </main>
    <footer>
        <p>&copy; {{ now.Format "2006" }} {{ .Site.Author.name }}</p>
    </footer>
</body>
</html>

<!-- ## Explore

{{< cards >}}
  {{< card link="docs" title="Docs" icon="book-open" >}}
  {{< card link="about" title="About" icon="user" >}}
{{< /cards >}}

## Documentation -->


