from typing import Optional

from data.stackexchange.generate_unix import main as generate_unix_tasks, download_and_extract_dataset, parse_posts_xml, extract_questions_from_data
from data.commons import (
    create_task_from_dockerfiles_and_questions,
    finalize_dataset_output,
    upload_tasks_to_hf,
    BaseDataGenerator,
    GenerationContext,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
)
from data.claude_docker.utils import gpt_convert_prompt_to_dockerfile_batched

class ClaudeDockerGenerator(BaseDataGenerator):
    parser_description = "Generate StackExchange-based docker tasks"

    def add_arguments(self, parser):
        parser.add_argument(
            "--source-url",
            type=str,
            default="https://archive.org/download/stackexchange/unix.stackexchange.com.7z",
            help="StackExchange archive URL",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=10,
            help="Number of questions to convert",
        )

    def run_task_generation(
        self,
        request: GenerationRequest,
        context: GenerationContext,
    ) -> GenerationResult:
        args = context.args
        effective_limit = self._resolve_limit(request) or args.limit
        posts_xml_path = download_and_extract_dataset(args.source_url)
        questions_data = parse_posts_xml(posts_xml_path)
        questions = extract_questions_from_data(questions_data)[: effective_limit]
        dockerfiles = gpt_convert_prompt_to_dockerfile_batched(questions)
        final_dataset_dir = create_task_from_dockerfiles_and_questions(dockerfiles, questions)
        output_path = finalize_dataset_output(final_dataset_dir, getattr(args, "output_dir", None))
        dataset_path = str(output_path)

        uploaded_repo: Optional[str] = None
        if context.args.target_repo and not getattr(context.args, "no_upload", False):
            upload_tasks_to_hf(dataset_path, context.args.target_repo)
            uploaded_repo = context.args.target_repo

        return GenerationResult(
            dataset_path=dataset_path,
            status=GenerationStatus.SUCCESS,
            uploaded_repo=uploaded_repo,
        )


def main() -> None:
    ClaudeDockerGenerator().run()


if __name__ == "__main__":
    main()
