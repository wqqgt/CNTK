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

    const std::wstring TrainingSession::s_checkpointIndex = L"CheckpointIndex";
    const std::wstring TrainingSession::s_trainingMinibatchSource = L"TrainingMinibatchSource";

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
        m_saveAllCheckpoints(saveAllCheckpoints)
    {
        if (!trainingSource)
            InvalidArgument("Minibatch source is not allowed to be null.");
        if (!trainer)
            InvalidArgument("Trainer is not allowed to be null.");
        if(modelInputToMinibatchSourceStream.empty())
            InvalidArgument("Input mapping is not allowed to be empty.");
        if (m_checkPointFileName.empty() && checkpointFrequencyInSamples != 0)
            InvalidArgument("Checkpoint file name is not allowed to be empty.");

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
        size_t workerRank = 0, numberOfWorkers = 1;
        size_t lastCheckpointSamplesSeen = 0;

        // Let's try to restore if required.
        if (m_restoreFromCheckpointIfExists)
            Restore();

        // Calculate how many local samples we are allowed to consume:
        const size_t maxLocalSamples = m_maxNumberOfSamples / m_numberOfWorkers;
        size_t localNumSamplesSeen = 0;

        while (shouldTrain)
        {
            // Check if we are operating in distributed mode.
            if (m_parallelAfterSamples >= m_trainer->TotalNumberOfSamplesSeen())
            {
                numberOfWorkers = m_numberOfWorkers;
                workerRank = m_workerRank;
            }

            // Get the minibatch
            size_t mbSize = GetMinibatchSize();
            minibatch.clear();
            if(localNumSamplesSeen < maxLocalSamples)
            {
                auto minibatchData = m_trainingSource->GetNextMinibatch(0 /*numberOfSequences*/, mbSize, numberOfWorkers, workerRank, computeDevice);
                if (!minibatchData.empty())
                {
                    for (auto v : m_modelInputToMinibatchSourceStream)
                        minibatch.insert({ v.first, minibatchData[v.second].data });

                    // Updating number of local samples seen.
                    localNumSamplesSeen += minibatchData.begin()->second.numberOfSamples;
                }
            }

            // Train on the minibatch
            OnMinibatchStart();
            shouldTrain = m_trainer->TrainMinibatch(minibatch, computeDevice);
            OnMinibatchEnd();

            // Check whether to create a checkpoint
            if (m_checkpointFrequencyinSamples > 0)
            {
                size_t checkpointIndex = m_trainer->TotalNumberOfSamplesSeen() / m_checkpointFrequencyinSamples;
                if (checkpointIndex > m_currentCheckpointIndex)
                {
                    m_currentCheckpointIndex = checkpointIndex;
                    SaveCheckpoint(false);
                    lastCheckpointSamplesSeen = m_trainer->TotalNumberOfSamplesSeen();
                }
            }
        }

        if (m_checkpointFrequencyinSamples > 0)
        {
            // Save the last checkpoint.
            SaveCheckpoint(true);
        }
    }

    void TrainingSession::RestoreFromCheckpoint(const std::wstring& checkpointFileName)
    {
        Dictionary externalState = m_trainer->RestoreFromCheckpoint(checkpointFileName);
        m_currentCheckpointIndex = externalState[s_checkpointIndex].Value<size_t>();
        m_trainingSource->RestoreFromCheckpoint(externalState[s_trainingMinibatchSource].Value<Dictionary>());
    }

    void TrainingSession::SaveCheckpoint(bool last)
    {
        OnCheckpointStart();
        Dictionary externalState;
        externalState[s_checkpointIndex] = m_currentCheckpointIndex;
        externalState[s_trainingMinibatchSource] = m_trainingSource->GetCheckpointState();

        wstring checkpointFile = m_checkPointFileName;
        if (m_saveAllCheckpoints && !last)
            checkpointFile += std::to_wstring(m_currentCheckpointIndex);
        m_trainer->SaveCheckpoint(m_checkPointFileName, externalState);
        OnCheckpointEnd();
    }

    void TrainingSession::Restore()
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
        this->RestoreFromCheckpoint(candidate);
    }
}
