-- Complete Schema for OT-Agents Registration System
-- Run this file to set up all required tables

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==================== DATASETS TABLE ====================
-- Stores dataset metadata for both HuggingFace and local datasets
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    created_by TEXT NOT NULL,
    creation_location TEXT NOT NULL,
    creation_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    generation_start TIMESTAMP WITH TIME ZONE,
    generation_end TIMESTAMP WITH TIME ZONE,
    data_location TEXT NOT NULL,
    generation_parameters JSONB NOT NULL,
    generation_status TEXT,
    dataset_type TEXT NOT NULL CHECK (dataset_type IN ('SFT', 'RL')),
    data_generation_hash TEXT,
    hf_fingerprint TEXT,
    hf_commit_hash TEXT,
    num_tasks INTEGER,
    last_modified TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for datasets
CREATE INDEX idx_datasets_name ON datasets(name);
CREATE INDEX idx_datasets_created_by ON datasets(created_by);
CREATE INDEX idx_datasets_dataset_type ON datasets(dataset_type);
CREATE INDEX idx_datasets_creation_time ON datasets(creation_time DESC);

-- ==================== AGENTS TABLE ====================
-- Stores evaluation agent metadata
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    agent_version_hash CHAR(64),
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for agents
CREATE INDEX idx_agents_name ON agents(name);
CREATE INDEX idx_agents_agent_version_hash ON agents(agent_version_hash);
CREATE INDEX idx_agents_updated_at ON agents(updated_at DESC);

-- ==================== MODELS TABLE ====================
-- Stores ML model metadata and training information
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    base_model_id UUID REFERENCES models(id),
    created_by TEXT NOT NULL,
    creation_location TEXT NOT NULL,
    creation_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    dataset_id UUID REFERENCES datasets(id),
    is_external BOOLEAN NOT NULL DEFAULT false,
    weights_location TEXT NOT NULL,
    wandb_link TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Training result related fields
    training_start TIMESTAMP WITH TIME ZONE NOT NULL,
    training_end TIMESTAMP WITH TIME ZONE,
    training_parameters JSONB NOT NULL,
    training_status TEXT,
    agent_id UUID REFERENCES agents(id) NOT NULL,
    training_type TEXT CHECK (training_type IN ('SFT', 'RL')),
    traces_location_s3 TEXT,
    description TEXT
);

-- Indexes for models
CREATE INDEX idx_models_name ON models(name);
CREATE INDEX idx_models_created_by ON models(created_by);
CREATE INDEX idx_models_agent_id ON models(agent_id);
CREATE INDEX idx_models_dataset_id ON models(dataset_id);
CREATE INDEX idx_models_base_model_id ON models(base_model_id);
CREATE INDEX idx_models_training_type ON models(training_type);
CREATE INDEX idx_models_creation_time ON models(creation_time DESC);
CREATE INDEX idx_models_training_start ON models(training_start DESC);

-- ==================== BENCHMARKS TABLE ====================
-- Stores evaluation benchmark metadata
CREATE TABLE IF NOT EXISTS benchmarks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    benchmark_version_hash CHAR(64),
    is_external BOOLEAN NOT NULL DEFAULT false,
    external_link TEXT,
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for benchmarks
CREATE INDEX idx_benchmarks_name ON benchmarks(name);
CREATE INDEX idx_benchmarks_benchmark_version_hash ON benchmarks(benchmark_version_hash);
CREATE INDEX idx_benchmarks_is_external ON benchmarks(is_external);
CREATE INDEX idx_benchmarks_updated_at ON benchmarks(updated_at DESC);

-- ==================== UPDATE TRIGGERS ====================
-- Auto-update timestamps on modification

-- Datasets trigger
CREATE OR REPLACE FUNCTION update_datasets_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_datasets_updated_at_trigger
BEFORE UPDATE ON datasets
FOR EACH ROW
EXECUTE FUNCTION update_datasets_updated_at();

-- Agents trigger
CREATE OR REPLACE FUNCTION update_agents_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_agents_updated_at_trigger
BEFORE UPDATE ON agents
FOR EACH ROW
EXECUTE FUNCTION update_agents_updated_at();

-- Models trigger
CREATE OR REPLACE FUNCTION update_models_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_models_updated_at_trigger
BEFORE UPDATE ON models
FOR EACH ROW
EXECUTE FUNCTION update_models_updated_at();

-- Benchmarks trigger
CREATE OR REPLACE FUNCTION update_benchmarks_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_benchmarks_updated_at_trigger
BEFORE UPDATE ON benchmarks
FOR EACH ROW
EXECUTE FUNCTION update_benchmarks_updated_at();

-- ==================== DOCUMENTATION ====================
-- Add helpful comments

-- Datasets table
COMMENT ON TABLE datasets IS 'Table storing dataset metadata for SFT and RL training';
COMMENT ON COLUMN datasets.dataset_type IS 'Type of dataset: SFT (Supervised Fine-Tuning) or RL (Reinforcement Learning)';
COMMENT ON COLUMN datasets.hf_fingerprint IS 'HuggingFace dataset fingerprint for tracking versions';
COMMENT ON COLUMN datasets.hf_commit_hash IS 'HuggingFace repository commit hash';
COMMENT ON COLUMN datasets.num_tasks IS 'Number of tasks/examples in the dataset';

-- Agents table
COMMENT ON TABLE agents IS 'Table storing evaluation agent metadata';
COMMENT ON COLUMN agents.name IS 'Name of the agent';
COMMENT ON COLUMN agents.agent_version_hash IS 'SHA-256 hash of the agent version (64 characters)';
COMMENT ON COLUMN agents.description IS 'Description of the agent and its capabilities';
COMMENT ON COLUMN agents.updated_at IS 'Timestamp when the agent was last updated';

-- Models table
COMMENT ON TABLE models IS 'Table storing ML model metadata and training information';
COMMENT ON COLUMN models.training_type IS 'Type of training: SFT (Supervised Fine-Tuning) or RL (Reinforcement Learning)';
COMMENT ON COLUMN models.is_external IS 'Whether this model is external (e.g., from HuggingFace)';
COMMENT ON COLUMN models.training_parameters IS 'JSON containing all training hyperparameters and configuration';

-- Benchmarks table
COMMENT ON TABLE benchmarks IS 'Table storing evaluation benchmark metadata';
COMMENT ON COLUMN benchmarks.name IS 'Name of the benchmark';
COMMENT ON COLUMN benchmarks.benchmark_version_hash IS 'SHA-256 hash of the benchmark version (64 characters)';
COMMENT ON COLUMN benchmarks.is_external IS 'Whether this benchmark is external (not hosted internally)';
COMMENT ON COLUMN benchmarks.external_link IS 'Link to external benchmark if applicable';
COMMENT ON COLUMN benchmarks.description IS 'Description of the benchmark and its purpose';
COMMENT ON COLUMN benchmarks.updated_at IS 'Timestamp when the benchmark was last updated';

-- ==================== SANDBOX TASKS TABLE ====================
-- Stores sandbox task definitions with checksum-based deduplication
CREATE TABLE IF NOT EXISTS sandbox_tasks (
    checksum TEXT NOT NULL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    source TEXT,
    name TEXT NOT NULL,
    instruction TEXT NOT NULL DEFAULT '',
    agent_timeout_sec NUMERIC NOT NULL,
    verifier_timeout_sec NUMERIC NOT NULL,
    git_url TEXT,
    git_commit_id TEXT,
    path TEXT NOT NULL,
    CONSTRAINT sandbox_task_source_name_key UNIQUE (source, name)
);

-- Indexes for sandbox_tasks
CREATE INDEX idx_sandbox_tasks_source_name ON sandbox_tasks(source, name);
CREATE INDEX idx_sandbox_tasks_name ON sandbox_tasks(name);
CREATE INDEX idx_sandbox_tasks_created_at ON sandbox_tasks(created_at DESC);

-- ==================== SANDBOX BENCHMARK TASKS TABLE ====================
-- Many-to-many relationship between benchmarks and tasks
CREATE TABLE IF NOT EXISTS sandbox_benchmark_tasks (
    benchmark_id UUID REFERENCES benchmarks(id) NOT NULL,
    benchmark_name TEXT NOT NULL,
    benchmark_version_hash CHAR(64) NOT NULL,
    task_checksum TEXT REFERENCES sandbox_tasks(checksum) NOT NULL,
    CONSTRAINT sandbox_benchmark_id_task_checksum UNIQUE (benchmark_id, task_checksum)
);

-- Indexes for sandbox_benchmark_tasks
CREATE INDEX idx_sandbox_benchmark_tasks_benchmark_id ON sandbox_benchmark_tasks(benchmark_id);
CREATE INDEX idx_sandbox_benchmark_tasks_task_checksum ON sandbox_benchmark_tasks(task_checksum);

-- ==================== SANDBOX JOBS TABLE ====================
-- Stores evaluation job/run metadata
CREATE TABLE IF NOT EXISTS sandbox_jobs (
    id UUID NOT NULL PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    job_name TEXT NOT NULL,
    username TEXT NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE,
    ended_at TIMESTAMP WITH TIME ZONE,
    git_commit_id TEXT,
    package_version TEXT,
    n_trials INTEGER NOT NULL,
    config JSONB NOT NULL,
    metrics JSONB,
    stats JSONB,
    agent_id UUID REFERENCES agents(id) NOT NULL,
    model_id UUID REFERENCES models(id) NOT NULL,
    benchmark_id UUID REFERENCES benchmarks(id) NOT NULL,
    n_rep_eval INTEGER NOT NULL,
    hf_traces_link TEXT,
    job_status JOB_STATUS_ENUM NOT NULL DEFAULT 'Started',

    -- Ensure at least one version identifier exists
    CONSTRAINT sandbox_job_version_check CHECK (
        git_commit_id IS NOT NULL OR package_version IS NOT NULL
    ),
    CONSTRAINT sandbox_job_agent_model_benchmark_key UNIQUE (agent_id, model_id, benchmark_id)
);

-- Indexes for sandbox_jobs
CREATE INDEX idx_sandbox_jobs_job_name ON sandbox_jobs(job_name);
CREATE INDEX idx_sandbox_jobs_username ON sandbox_jobs(username);
CREATE INDEX idx_sandbox_jobs_agent_id ON sandbox_jobs(agent_id);
CREATE INDEX idx_sandbox_jobs_model_id ON sandbox_jobs(model_id);
CREATE INDEX idx_sandbox_jobs_benchmark_id ON sandbox_jobs(benchmark_id);
CREATE INDEX idx_sandbox_jobs_created_at ON sandbox_jobs(created_at DESC);
CREATE INDEX idx_sandbox_jobs_started_at ON sandbox_jobs(started_at DESC);

-- ==================== SANDBOX TRIALS TABLE ====================
-- Stores individual trial/task execution results
CREATE TABLE IF NOT EXISTS sandbox_trials (
    id UUID NOT NULL PRIMARY KEY DEFAULT uuid_generate_v4(),
    trial_name TEXT NOT NULL,
    trial_uri TEXT NOT NULL,
    job_id UUID REFERENCES sandbox_jobs(id),
    task_checksum TEXT REFERENCES sandbox_tasks(checksum) NOT NULL,
    reward NUMERIC,

    -- Detailed timing information
    started_at TIMESTAMP WITH TIME ZONE,
    ended_at TIMESTAMP WITH TIME ZONE,
    environment_setup_started_at TIMESTAMP WITH TIME ZONE,
    environment_setup_ended_at TIMESTAMP WITH TIME ZONE,
    agent_setup_started_at TIMESTAMP WITH TIME ZONE,
    agent_setup_ended_at TIMESTAMP WITH TIME ZONE,
    agent_execution_started_at TIMESTAMP WITH TIME ZONE,
    agent_execution_ended_at TIMESTAMP WITH TIME ZONE,
    verifier_started_at TIMESTAMP WITH TIME ZONE,
    verifier_ended_at TIMESTAMP WITH TIME ZONE,

    config JSONB NOT NULL,
    exception_info JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Indexes for sandbox_trials
CREATE INDEX idx_sandbox_trials_trial_name ON sandbox_trials(trial_name);
CREATE INDEX idx_sandbox_trials_job_id ON sandbox_trials(job_id);
CREATE INDEX idx_sandbox_trials_task_checksum ON sandbox_trials(task_checksum);
CREATE INDEX idx_sandbox_trials_created_at ON sandbox_trials(created_at DESC);
CREATE INDEX idx_sandbox_trials_started_at ON sandbox_trials(started_at DESC);

-- ==================== SANDBOX TRIAL MODEL USAGE TABLE ====================
-- Tracks model usage (tokens) during trial execution
CREATE TABLE IF NOT EXISTS sandbox_trial_model_usage (
    trial_id UUID REFERENCES sandbox_trials(id) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    model_id UUID REFERENCES models(id) NOT NULL,
    model_provider TEXT NOT NULL,
    n_input_tokens INTEGER,
    n_output_tokens INTEGER,
    PRIMARY KEY (trial_id, model_id, model_provider)
);

-- Indexes for sandbox_trial_model_usage
CREATE INDEX idx_sandbox_trial_model_usage_trial_id ON sandbox_trial_model_usage(trial_id);
CREATE INDEX idx_sandbox_trial_model_usage_model_id ON sandbox_trial_model_usage(model_id);

-- ==================== SANDBOX DOCUMENTATION ====================
-- Add helpful comments for sandbox tables

-- Sandbox tasks table
COMMENT ON TABLE sandbox_tasks IS 'Sandbox task definitions with checksum-based deduplication';
COMMENT ON COLUMN sandbox_tasks.checksum IS 'SHA-256 checksum of task content for deduplication';
COMMENT ON COLUMN sandbox_tasks.source IS 'Source of the task (e.g., repository name)';
COMMENT ON COLUMN sandbox_tasks.name IS 'Name of the task';
COMMENT ON COLUMN sandbox_tasks.instruction IS 'Task instruction text (may be empty)';
COMMENT ON COLUMN sandbox_tasks.agent_timeout_sec IS 'Timeout in seconds for agent execution';
COMMENT ON COLUMN sandbox_tasks.verifier_timeout_sec IS 'Timeout in seconds for verifier execution';

-- Sandbox benchmark tasks table
COMMENT ON TABLE sandbox_benchmark_tasks IS 'Many-to-many relationship between benchmarks and sandbox tasks';
COMMENT ON COLUMN sandbox_benchmark_tasks.benchmark_id IS 'Reference to benchmark';
COMMENT ON COLUMN sandbox_benchmark_tasks.task_checksum IS 'Reference to sandbox task';

-- Sandbox jobs table
COMMENT ON TABLE sandbox_jobs IS 'Sandbox evaluation jobs/runs';
COMMENT ON COLUMN sandbox_jobs.job_name IS 'Name of the evaluation job';
COMMENT ON COLUMN sandbox_jobs.agent_id IS 'Reference to the agent used';
COMMENT ON COLUMN sandbox_jobs.model_id IS 'Reference to the model used';
COMMENT ON COLUMN sandbox_jobs.benchmark_id IS 'Reference to the benchmark evaluated';

-- Sandbox trials table
COMMENT ON TABLE sandbox_trials IS 'Individual sandbox trial executions';
COMMENT ON COLUMN sandbox_trials.trial_name IS 'Name of the trial';
COMMENT ON COLUMN sandbox_trials.job_id IS 'Reference to parent job';
COMMENT ON COLUMN sandbox_trials.task_checksum IS 'Reference to task that was executed';
COMMENT ON COLUMN sandbox_trials.reward IS 'Reward/score achieved in this trial';

-- Sandbox trial model usage table
COMMENT ON TABLE sandbox_trial_model_usage IS 'Model usage tracking for trials';
COMMENT ON COLUMN sandbox_trial_model_usage.trial_id IS 'Reference to trial';
COMMENT ON COLUMN sandbox_trial_model_usage.model_id IS 'Reference to model used';
COMMENT ON COLUMN sandbox_trial_model_usage.model_provider IS 'Provider of the model (e.g., openai, anthropic)';