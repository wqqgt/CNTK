//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "CNTKLibrary.h"
#include "fileutil.h"

namespace CNTK
{
    using namespace std;

    const static std::wstring s_checkpointIndex = L"CheckpointIndex";
    const static std::wstring s_trainingMinibatchSource = L"TrainingMinibatchSource";

    TrainingSessionPtr CreateBasicTrainingSession(
        const MinibatchSourcePtr& trainingSource,
        const TrainerPtr& trainer,
        const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
        const MinibatchSizeSchedule& minibatchSizeSchedule,
        size_t checkpointFrequencyinSamples,
        const std::wstring& checkPointFileName)
    {
        return MakeSharedObject<TrainingSession>(trainingSource,
            trainer,
            modelInputToMinibatchSourceStream,
            minibatchSizeSchedule,
            checkpointFrequencyinSamples,
            checkPointFileName);
    }

    TrainingSession::TrainingSession(
        const MinibatchSourcePtr& trainingSource,
        const TrainerPtr& trainer,
        const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
        const MinibatchSizeSchedule& schedule,
        size_t checkpointFrequencyInSamples,
        const std::wstring& checkPointFileName,
        const MinibatchSourcePtr& crossValidationSource = nullptr,
        size_t crossValidationFrequencyInSamples = 0,
        bool restoreFromCheckpointIfExists,
        bool saveAllCheckpoints,
        size_t maxNumberOfSamples) :
        m_trainingSource(trainingSource),
        m_trainer(trainer),
        m_modelInputToMinibatchSourceStream(modelInputToMinibatchSourceStream),
        m_checkpointFrequencyinSamples(checkpointFrequencyInSamples),
        m_checkPointFileName(checkPointFileName),
        m_currentCheckpointIndex(0),
        m_parallelAfterSamples(0),
        m_workerRank(0),
        m_numberOfWorkers(1),
        m_minibatchSizeSchedule(schedule),
        m_maxNumberOfSamples(maxNumberOfSamples),
        m_restoreFromCheckpointIfExists(restoreFromCheckpointIfExists),
        m_saveAllCheckpoints(saveAllCheckpoints),
        m_crossValidationFrequencyInSamples(crossValidationFrequencyInSamples),
        m_crossValidationSource(crossValidationSource),
        m_currentCrossValidationIndex(0)
    {
        if (!trainingSource)
            InvalidArgument("Training minibatch source is not allowed to be null.");
        if (!trainer)
            InvalidArgument("Trainer is not allowed to be null.");
        if(modelInputToMinibatchSourceStream.empty())
            InvalidArgument("Input mapping is not allowed to be empty.");
        if (m_checkPointFileName.empty() && checkpointFrequencyInSamples != 0)
            InvalidArgument("Checkpoint file name is not allowed to be empty.");
        if (!m_crossValidationSource && crossValidationFrequencyInSamples != 0)
            InvalidArgument("Cross validation minibatch source is not allowed to be empty.");

        // Let's calculate the warm up period the distributed learners may need.
        // We will take the maximum warm up period required.
        auto learners = trainer->ParameterLearners();
        m_parallelAfterSamples = 0;
        for (const auto& l: learners)
        {
            auto distributed = std::dynamic_pointer_cast<DistributedLearner>(l);
            if (distributed)
            {
                m_parallelAfterSamples = std::max(m_parallelAfterSamples, distributed->ParallelizationAfter());
                m_workerRank = distributed->GetCommunicator()->CurrentWorker().m_globalRank;
                m_numberOfWorkers = distributed->GetCommunicator()->Workers().size();
            }
        }
    }

    inline bool isNumber(const std::wstring& s)
    {
        return !s.empty() && 
            find_if(s.begin(), s.end(), [](wchar_t c) { return !isdigit(c); }) == s.end();
    }

    void TrainingSession::Train(const DeviceDescriptor& computeDevice)
    {
        std::unordered_map<Variable, ValuePtr> minibatch;
        bool shouldTrain = m_maxNumberOfSamples > 0;

        // Let's try to restore if required.
        if (m_restoreFromCheckpointIfExists)
            RestoreCheckpoint();

        // Main train loop.
        while (shouldTrain)
        {
            // Get next minibatch.
            if (m_trainer->TotalNumberOfSamplesSeen() < m_maxNumberOfSamples)
                GetTrainingMinibatch(minibatch, computeDevice);
            else
                minibatch.clear();

            // Train on the minibatch
            OnMinibatchStart();
            shouldTrain = m_trainer->TrainMinibatch(minibatch, computeDevice);
            OnMinibatchEnd();

            // Check whether to create a checkpoint
            PerformCheckPointIfNeeded();

            // Check whether to perform cross validation
            PerformCrossValidationIfNeeded();
        }

        if (m_checkpointFrequencyinSamples > 0)
        {
            // Always save the last checkpoint.
            SaveCheckpoint(true);
        }
    }

    void TrainingSession::CrossValidate(const DeviceDescriptor& computeDevice)
    {
        std::unordered_map<Variable, ValuePtr> minibatch;

        double accumulatedError = 0;
        size_t numberOfMinibatches = 0;
        while(GetCrossValidationMinibatch(minibatch, computeDevice), !minibatch.empty())
        {
            accumulatedError += m_trainer->TestMinibatch(minibatch, computeDevice);
            numberOfMinibatches++;
        }

        OnCrossValidationEnd(m_currentCrossValidationIndex, accumulatedError/numberOfMinibatches);
    }

    inline void TrainingSession::PerformCheckPointIfNeeded()
    {
        if (m_checkpointFrequencyinSamples == 0)
            return;

        size_t checkpointIndex = m_trainer->TotalNumberOfSamplesSeen() / m_checkpointFrequencyinSamples;
        if (checkpointIndex <= m_currentCheckpointIndex)
            return; // Nothing to do.

        // Perform the checkpoint.
        m_currentCheckpointIndex = checkpointIndex;
        SaveCheckpoint(false);
    }

    inline void TrainingSession::PerformCrossValidationIfNeeded()
    {
        if (m_crossValidationFrequencyInSamples == 0)
            return;

        size_t crossValidationIndex = m_trainer->TotalNumberOfSamplesSeen() / m_crossValidationFrequencyInSamples;
        if (crossValidationIndex <= m_currentCrossValidationIndex)
            return; // Nothing to do.

        // Perform cross validation
        m_currentCrossValidationIndex = crossValidationIndex;
        CrossValidate();
    }

    void TrainingSession::GetTrainingMinibatch(std::unordered_map<Variable, ValuePtr>& minibatch, const DeviceDescriptor& computeDevice)
    {
        size_t workerRank = 0, numberOfWorkers = 1;

        // Check if we are operating in distributed mode.
        if (m_parallelAfterSamples >= m_trainer->TotalNumberOfSamplesSeen())
        {
            numberOfWorkers = m_numberOfWorkers;
            workerRank = m_workerRank;
        }

        GetNextMinibatch(m_trainingSource, minibatch, workerRank, numberOfWorkers, computeDevice);
    }

    void TrainingSession::GetCrossValidationMinibatch(std::unordered_map<Variable, ValuePtr>& minibatch, const DeviceDescriptor& computeDevice)
    {
        // TODO: Support distributed cross-validation, when TestMinibatch supports it.
        GetNextMinibatch(m_crossValidationSource, minibatch, 0, 1, computeDevice);
    }

    void TrainingSession::GetNextMinibatch(const MinibatchSourcePtr& source, std::unordered_map<Variable, ValuePtr>& minibatch, size_t workerRank, size_t numberOfWorkers, const DeviceDescriptor& computeDevice)
    {
        size_t mbSize = GetMinibatchSize();
        minibatch.clear();
        auto minibatchData = source->GetNextMinibatch(0 /*numberOfSequences*/, mbSize, numberOfWorkers, workerRank, computeDevice);
        if (minibatchData.empty())
            return;

        for (auto v : m_modelInputToMinibatchSourceStream)
            minibatch.insert({ v.first, minibatchData[v.second].data });
    }

    void TrainingSession::RestoreFromCheckpoint(const std::wstring& checkpointFileName)
    {
        Dictionary externalState = m_trainer->RestoreFromCheckpoint(checkpointFileName);
        m_currentCheckpointIndex = externalState[s_checkpointIndex].Value<size_t>();
        m_trainingSource->RestoreFromCheckpoint(externalState[s_trainingMinibatchSource].Value<Dictionary>());
    }

    void TrainingSession::SaveCheckpoint(bool last)
    {
        OnCheckpointStart(m_currentCheckpointIndex);
        Dictionary externalState;
        externalState[s_checkpointIndex] = m_currentCheckpointIndex;
        externalState[s_trainingMinibatchSource] = m_trainingSource->GetCheckpointState();

        wstring checkpointFile = m_checkPointFileName;
        if (m_saveAllCheckpoints && !last)
            checkpointFile += std::to_wstring(m_currentCheckpointIndex);
        m_trainer->SaveCheckpoint(m_checkPointFileName, externalState);
        OnCheckpointEnd(m_currentCheckpointIndex);
    }

    void TrainingSession::RestoreCheckpoint()
    {
        // Make sure the intermediate directories exist, so no need for further checks.
        msra::files::make_intermediate_dirs(m_checkPointFileName);

        // Best or single checkpoint found - simply resoring from it.
        if (boost::filesystem::exists(m_checkPointFileName))
        {
            this->RestoreFromCheckpoint(m_checkPointFileName);
            return;
        }

        // If not - let's check whether there are other possible candidates to restore from.
        using namespace boost::filesystem;

        int maxValue = -1;
        wstring candidate;
        auto d = wpath(m_checkPointFileName).parent_path();
        for (directory_iterator itr(d); itr != directory_iterator(); ++itr)
        {
            if (!is_regular_file(itr->status()) ||
                !boost::starts_with(itr->path().c_str(), m_checkPointFileName))
            {
                continue;
            }

            std::wstring filePath = itr->path().c_str();
            auto suffix = filePath.substr(m_checkPointFileName.size());
            if (!isNumber(suffix) || !boost::filesystem::exists(filePath + L".ckp"))
            {
                continue;
            }

            auto expectedNumber = msra::strfun::utf8(suffix);
            char* tmp;
            int value = strtol(expectedNumber.c_str(), &tmp, 10);
            assert(tmp == expectedNumber.c_str() + expectedNumber.size());

            if (value > maxValue)
            {
                // Found a better candidate.
                maxValue = value;
                candidate = filePath;
            }
        }

        // Restoring from the candidate.
        if (!candidate.empty())
            this->RestoreFromCheckpoint(candidate);
    }
}
