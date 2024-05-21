import torch
import torch.nn as nn
import torch.nn.functional as F


def get_logits(self, im_embedding, sp_embedding, temperature=15.5):  # for testing
    im_embedding = F.normalize(im_embedding, dim=-1, eps=1e-3)
    sp_embedding = F.normalize(sp_embedding, dim=-1, eps=1e-3)

    im_cos_matrix = temperature * im_embedding @ sp_embedding.T
    return im_cos_matrix, im_cos_matrix.T


class CLIPLoss(nn.Module):
    def forward(self, im_emb, sp_emb, temperature, output_as_dict=False):
        """
        @brief Calculates the contrastive loss between image and spectrum embeddings using cosine similarity.

        This method normalizes the image and spectrum embeddings and then computes their cosine similarity
        matrix scaled by a specified temperature factor. It computes the cross-entropy loss using this similarity
        matrix against a set of labels that are simply the indices of the batch, aiming to match each item with itself
        across the two modalities. The function supports returning the loss either as a scalar or as a dictionary.

        @param im_emb Tensor containing embeddings of images, shape should be [batch_size, embedding_size].
        @param sp_emb Tensor containing embeddings of spectra, shape should be [batch_size, embedding_size].
        @param temperature A scaling factor to adjust the sharpness of the softmax probabilities derived from the cosine similarities.
        @param output_as_dict A boolean flag to determine whether to return the loss as a dictionary or as a scalar.
                            If True, the loss is returned in a dictionary under the key 'overall_contrastive_loss'.

        @return Either a dictionary containing the loss if output_as_dict is True, or a scalar tensor representing the loss.
        """
        # Normalise embeddings and compute the cosine similarity with scaling
        norm_im_emb = F.normalize(im_emb, p=2, dim=-1, eps=1e-3)
        norm_sp_emb = F.normalize(sp_emb, p=2, dim=-1, eps=1e-3)

        im_cos_matrix = temperature * torch.matmul(norm_im_emb, norm_sp_emb.T)
        sp_cos_matrix = im_cos_matrix.T
        # Create sequence of indices for batch samples to use as labels
        labels = torch.arange(
            im_cos_matrix.size(0), device=im_emb.device, dtype=torch.long
        )

        # Calculate the average of both computed losses
        total_loss = 0.5 * (
            F.cross_entropy(im_cos_matrix, labels)
            + F.cross_entropy(sp_cos_matrix, labels)
        )

        if output_as_dict:
            return {"overall_contrastive_loss": total_loss}
        return total_loss

        # def get_cosine_matrix(self, im_embedding, sp_embedding, temperature=15.5):

    #     im_embedding = F.normalize(im_embedding, dim=-1, eps=1e-3)
    #     sp_embedding = F.normalize(sp_embedding, dim=-1, eps=1e-3)

    #     im_cos_matrix = temperature * torch.matmul(
    #         im_embedding, sp_embedding.T
    #     )  # ith image with jth spectrum
    #     return im_cos_matrix, im_cos_matrix.T

    # def forward(self, im_embedding, sp_embedding, temperature, output_dict=False):
    #     im_cos_matrix, sp_cos_matrix = self.get_cosine_matrix(
    #         im_embedding, sp_embedding, temperature
    #     )
    #     labels = torch.arange(
    #         im_cos_matrix.shape[0], device=im_embedding.device, dtype=torch.long
    #     )  # get labels for all the images in batch
    #     total_loss = (
    #         F.cross_entropy(im_cos_matrix, labels)
    #         + F.cross_entropy(sp_cos_matrix, labels)
    #     ) / 2
    #     return {"contrastive_loss": total_loss} if output_dict else total_loss
