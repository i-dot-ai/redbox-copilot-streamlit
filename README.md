# 📮 Redbox Copilot \[Streamlit app\]

> [!IMPORTANT]
> Incubation Project: This project is an incubation project; as such, we DON’T recommend using it in any critical use case. This project is in active development and a work in progress. This project may one day Graduate, in which case this disclaimer will be removed.

Redbox Copilot is a retrieval augmented generation (RAG) app that uses GenAI to chat with and summarise civil service documents. It's designed to handle a variety of administrative sources, such as letters, briefings, minutes, and speech transcripts.

- **Better retrieval**. Redbox Copilot increases organisational memory by indexing documents
- **Faster, accurate summarisation**. Redbox Copilot can summarise reports read months ago, supplement them with current work, and produce a first draft that lets civil servants focus on what they do best

This repo contains a [Streamlit frontend](https://streamlit.io) for the [core Redbox project](https://github.com/i-dot-ai/redbox-copilot).

# Contributing

Development is done through [dev containers](https://code.visualstudio.com/docs/devcontainers/create-dev-container) to control for varying dependency requirements across operating systems for libraries like [torch](https://pytorch.org/get-started/locally/).

We use [poetry](https://python-poetry.org) to manage requirements, and these assume Linux.

We welcome contributions to this project. Please see the [CONTRIBUTING.md](./CONTRIBUTING.md) file for more information.

## First time setup

Clone to repo to your local machine.

### Dev container

Before you build the container, get the absolute path to your repo:

```console
pwd
```

Copy this path.

Build the container. When prompted, give the path you copied. See the [docker-outside-of-docker documentation](https://github.com/devcontainers/features/tree/main/src/docker-outside-of-docker) for information and troubleshooting.

### Environment variables

Set up environment variables. Use `.env.example` to get started.

```console
cp .env.example .env
```

Setting one of `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is essential.

If you have issues with permissions, you may need to run `chmod 777 data/elastic/` to be able to write to the folder.

## Usage

We use [GNU make](https://www.gnu.org/software/make/manual/make.html) to help run the project. To see all commands:

```console
make
```

To start the app:

```console
make up
```

To stop the app:

```console
make down
```

The project can take up a lot of space in docker image and build caches. It may be necessary to clear these from time to time.

The following commands are potentially destrictive and so we don't give them easy make commands -- run them at your own risk.

Clear your docker image cache:

```console
docker image prune
```

Clear your docker build cache:

```console
docker builder prune
```

# License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

# Security

If you discover a security vulnerability within this project, please follow our [Security Policy](./SECURITY.md).
